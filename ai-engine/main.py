import cv2, base64, time, os, json
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import supervision as sv

from config import config
from detector import detector
from scorer import (is_after_hours, calculate_threat_score,
    apply_score_decay, get_status, check_cooldown, log_alert,
    update_critical_timer, critical_sustained)
from annotator import annotate_frame
from tracker import dwell_tracker
from motion import motion_analyzer

app = FastAPI(title='VisionIQ', version='3.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

os.makedirs('snapshots', exist_ok=True)
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")

clients     = []
ip_stream   = {"active": False, "url": "", "task": None}
executor    = ThreadPoolExecutor(max_workers=2)


# ── FRAME PROCESSOR (runs in thread executor) ─────────────
def process_one_frame(frame):
    detections   = detector.detect(frame)
    person_count = len(detections)
    h, w         = frame.shape[:2]

    motion = motion_analyzer.analyze(frame)

    zone_results       = []
    any_zone_triggered = False

    for zone_data in config.zones:
        triggered = False
        if person_count > 0:
            try:
                zone          = sv.PolygonZone(
                    polygon=np.array(zone_data['coords']),
                    frame_resolution_wh=(w, h)
                )
                in_zone_flags = zone.trigger(detections=detections)
                triggered     = bool(in_zone_flags.any())

                if detections.tracker_id is not None and triggered:
                    dwell_tracker.update(
                        detections.tracker_id,
                        zone_data['name'],
                        in_zone_flags
                    )
            except Exception as e:
                print("Zone error:", e)

        zone_results.append((zone_data['name'], triggered, zone_data['threat_level']))
        if triggered:
            any_zone_triggered = True

    if detections.tracker_id is not None:
        dwell_tracker.clear_lost_tracks(detections.tracker_id)

    loitering_ids   = list(dwell_tracker.loitering.keys())
    loitering_count = len(loitering_ids)

    after_hours = is_after_hours()
    raw_score   = calculate_threat_score(
        person_count, zone_results, after_hours, loitering_count, motion
    )
    score  = apply_score_decay(raw_score)
    status = get_status(score)

    update_critical_timer(status)

    snapshot_path   = None
    alert_triggered = False

    if status == 'CRITICAL' and critical_sustained() and check_cooldown():
        alert_triggered        = True
        config.last_alert_time = time.time()
        timestamp              = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_path          = f'snapshots/alert_{timestamp}.jpg'
        log_alert(score, status, person_count, snapshot_path, {
            'loitering_count': loitering_count,
            'zones_triggered': [z[0] for z in zone_results if z[1]],
            'running':         motion.get('running'),
            'panic':           motion.get('panic'),
            'abandoned':       motion.get('abandoned'),
        })
    elif status == 'SUSPICIOUS' and check_cooldown():
        config.last_alert_time = time.time()
        timestamp              = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_path          = f'snapshots/alert_{timestamp}.jpg'
        log_alert(score, status, person_count, snapshot_path, {
            'loitering_count': loitering_count,
        })

    annotated = annotate_frame(
        frame, detections, zone_results, score, status, loitering_ids, motion
    )

    if snapshot_path:
        cv2.imwrite(snapshot_path, annotated)
        print(f"Snapshot saved: {snapshot_path}")

    _, buffer     = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    annotated_b64 = base64.b64encode(buffer).decode()

    alerts = [
        {
            'timestamp':     e.get('timestamp', ''),
            'status':        e.get('status', ''),
            'score':         e.get('score', 0),
            'person_count':  e.get('person_count', 0),
            'snapshot_path': e.get('snapshot_path'),
        }
        for e in config.alert_log
        if e.get('status') in ('CRITICAL', 'SUSPICIOUS')
    ][:20]

    return {
        'annotated_base64':  annotated_b64,
        'person_count':      person_count,
        'threat_score':      round(score),
        'status':            status,
        'alert_triggered':   alert_triggered,
        'zone_triggered':    any_zone_triggered,
        'zone_results':      [{'name': z[0], 'triggered': z[1], 'threat_level': z[2]} for z in zone_results],
        'loitering_count':   loitering_count,
        'motion': {
            'running':   motion.get('running'),
            'panic':     motion.get('panic'),
            'abandoned': motion.get('abandoned'),
        },
        'after_hours':       after_hours,
        'snapshot_path':     snapshot_path,
        'alerts':            alerts,
    }


# ── BROADCAST ─────────────────────────────────────────────
async def broadcast(result):
    dead = []
    for client in clients:
        try:
            await client.send_text(json.dumps(result))
        except:
            dead.append(client)
    for d in dead:
        clients.remove(d)


# ── IP CAM LOOP ───────────────────────────────────────────
async def ip_cam_loop(url: str):
    print(f"Starting IP cam stream: {url}")
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"ERROR: Cannot open stream: {url}")
        ip_stream["active"] = False
        return

    print("IP cam connected!")
    loop       = asyncio.get_event_loop()
    fail_count = 0

    while ip_stream["active"]:
        ret, frame = cap.read()

        if not ret:
            fail_count += 1
            if fail_count > 10:
                print("Too many failures — stopping IP cam")
                break
            await asyncio.sleep(0.1)
            continue

        fail_count = 0
        frame  = cv2.resize(frame, (640, 480))

        # Run processing in thread so async loop stays free
        result = await loop.run_in_executor(executor, process_one_frame, frame)
        await broadcast(result)
        await asyncio.sleep(0.033)  # ~30 FPS

    cap.release()
    ip_stream["active"] = False
    print("IP cam stream stopped")


# ── ENDPOINTS ─────────────────────────────────────────────
class StreamRequest(BaseModel):
    url: str

@app.post("/start_stream")
async def start_stream(body: StreamRequest):
    if ip_stream["active"]:
        ip_stream["active"] = False
        await asyncio.sleep(0.3)

    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL required")

    ip_stream["active"] = True
    ip_stream["url"]    = url
    ip_stream["task"]   = asyncio.create_task(ip_cam_loop(url))
    return {"status": "started", "url": url}


@app.post("/stop_stream")
async def stop_stream():
    ip_stream["active"] = False
    if ip_stream["task"]:
        ip_stream["task"].cancel()
        ip_stream["task"] = None
    return {"status": "stopped"}


@app.get("/stream_status")
async def stream_status():
    return {"active": ip_stream["active"], "url": ip_stream["url"]}


@app.get('/health')
def health():
    return {
        'status':  'running',
        'model':   'YOLO11s',
        'mode':    config.mode,
        'ip_cam':  ip_stream["active"],
        'ip_url':  ip_stream["url"],
        'clients': len(clients),
    }


@app.post('/override_safe')
def override_safe():
    config.prev_score = 0.0
    log_alert(0, 'OVERRIDE', 0)
    return {'status': 'overridden'}


# ── WEBSOCKET ─────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print(f"Client connected — total: {len(clients)}")
    loop = asyncio.get_event_loop()

    try:
        while True:
            b64 = await websocket.receive_text()

            if ip_stream["active"]:
                continue

            try:
                img_data = base64.b64decode(b64)
                nparr    = np.frombuffer(img_data, np.uint8)
                frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                print(f"Frame decode error: {e}")
                continue

            # Run in thread executor — keeps async loop free
            result = await loop.run_in_executor(executor, process_one_frame, frame)
            await broadcast(result)

    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)
        print(f"Client disconnected — total: {len(clients)}")