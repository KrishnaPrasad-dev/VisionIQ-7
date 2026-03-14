import cv2, base64, time, os, json
import numpy as np
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

clients = []


def process_one_frame(frame):
    detections   = detector.detect(frame)
    person_count = len(detections)
    h, w         = frame.shape[:2]

    motion = motion_analyzer.analyze(frame)

    # ── ZONE CHECKS ──────────────────────────────────────
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
                    loiter_alerts = dwell_tracker.update(
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

    # ── SCORING ──────────────────────────────────────────
    after_hours = is_after_hours()
    raw_score   = calculate_threat_score(
        person_count, zone_results, after_hours, loitering_count, motion
    )
    score  = apply_score_decay(raw_score)
    status = get_status(score)

    # ── SNAPSHOT + ALERT (with cooldown) ─────────────────
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
        # Shared cooldown — SUSPICIOUS also uses last_alert_time
        config.last_alert_time = time.time()
        timestamp              = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_path          = f'snapshots/alert_{timestamp}.jpg'
        log_alert(score, status, person_count, snapshot_path, {
            'loitering_count': loitering_count,
        })

    # ── ANNOTATE ─────────────────────────────────────────
    annotated = annotate_frame(
        frame, detections, zone_results, score, status, loitering_ids, motion
    )

    # Save snapshot after annotation so it has bounding boxes
    if snapshot_path:
        cv2.imwrite(snapshot_path, annotated)
        print(f"Snapshot saved: {snapshot_path}")

    _, buffer     = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    annotated_b64 = base64.b64encode(buffer).decode()

    # Build alert list for dashboard (last 20, no SAFE)
    alerts = [
        {
            'timestamp':    e.get('timestamp', ''),
            'status':       e.get('status', ''),
            'score':        e.get('score', 0),
            'person_count': e.get('person_count', 0),
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


# ── WebSocket ─────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print(f"Client connected — total: {len(clients)}")

    try:
        while True:
            b64 = await websocket.receive_text()

            try:
                img_data = base64.b64decode(b64)
                nparr    = np.frombuffer(img_data, np.uint8)
                frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                print(f"Frame decode error: {e}")
                continue

            result = process_one_frame(frame)

            # Broadcast to all connected clients
            dead = []
            for client in clients:
                try:
                    await client.send_text(json.dumps(result))
                except:
                    dead.append(client)
            for d in dead:
                clients.remove(d)

    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)
        print(f"Client disconnected — total: {len(clients)}")