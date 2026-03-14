import cv2
import base64
import json
import asyncio
import websockets
import numpy as np
import threading
import time

WS_URL = "ws://localhost:8000/ws/stream"

# ── CHANGE SOURCE HERE ─────────────────────────────────────
SOURCE = 0   # ← change to your phone IP

latest_annotated = None
latest_stats     = {}
lock             = threading.Lock()
running          = True

print("=" * 50)
print("  VisionIQ — Fast Client")
print(f"  Source: {SOURCE}")
print("=" * 50)

async def sender(ws, cap):
    """Sends frames as fast as possible — doesn't wait for response."""
    while running:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.05)
            continue

        frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        b64 = base64.b64encode(buffer).decode('utf-8')

        try:
            await ws.send(b64)
        except:
            break

        await asyncio.sleep(0.033)  # ~30 FPS send rate

async def receiver(ws):
    """Receives results independently — no waiting on sender."""
    global latest_annotated, latest_stats

    frame_count = 0
    start_time  = time.time()

    while running:
        try:
            raw    = await ws.recv()
            result = json.loads(raw)

            ann_bytes = base64.b64decode(result['annotated_base64'])
            nparr     = np.frombuffer(ann_bytes, np.uint8)
            ann_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            with lock:
                latest_annotated = ann_frame
                latest_stats     = result

            frame_count += 1
            fps = frame_count / (time.time() - start_time)

            motion   = result.get('motion', {})
            flags    = []
            if motion.get('running'):            flags.append('RUNNING')
            if motion.get('panic'):              flags.append('PANIC')
            if motion.get('abandoned'):          flags.append('ABANDONED')
            if result.get('loitering_count', 0): flags.append(f"LOITERING:{result['loitering_count']}")
            flag_str = ' | ' + ' '.join(flags) if flags else ''

            print(
                f"People: {result['person_count']} | "
                f"Score: {result['threat_score']} | "
                f"Status: {result['status']} | "
                f"FPS: {fps:.1f}"
                f"{flag_str}"
            )

        except:
            break

async def stream():
    global running

    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {SOURCE}")
        return

    print("Source opened!")

    try:
        async with websockets.connect(WS_URL, max_size=10_000_000) as ws:
            print("Connected to AI engine!\n")

            # Run sender and receiver in parallel
            await asyncio.gather(
                sender(ws, cap),
                receiver(ws)
            )

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        running = False

def display_thread():
    global running
    while running:
        with lock:
            ann = latest_annotated.copy() if latest_annotated is not None else None
        if ann is not None:
            cv2.imshow('VisionIQ — AI Feed', ann)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        time.sleep(0.01)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    t = threading.Thread(target=display_thread, daemon=True)
    t.start()
    asyncio.run(stream())