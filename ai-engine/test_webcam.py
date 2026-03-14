import cv2
import base64
import json
import asyncio
import websockets
import numpy as np
import threading
import time

WS_URL = "ws://localhost:8000/ws"

latest_annotated = None
lock = threading.Lock()

async def ws_client():
    global latest_annotated
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    async with websockets.connect(WS_URL) as ws:
        print("Connected to WebSocket!")
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Send frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buffer).decode('utf-8')
            await ws.send(b64)

            # Receive result
            result = json.loads(await ws.recv())

            # Decode annotated frame
            ann_bytes = base64.b64decode(result['annotated_base64'])
            nparr = np.frombuffer(ann_bytes, np.uint8)
            ann_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            with lock:
                latest_annotated = ann_frame

            frame_count += 1
            fps = frame_count / (time.time() - start_time)
            print(f"People: {result['person_count']} | Score: {result['threat_score']} | Status: {result['status']} | FPS: {fps:.1f}")

            # Show windows
            cv2.imshow('Live Camera', frame)
            with lock:
                if latest_annotated is not None:
                    cv2.imshow('SmartEye AI', latest_annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

asyncio.run(ws_client())