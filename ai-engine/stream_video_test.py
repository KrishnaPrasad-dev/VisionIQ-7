import cv2
import base64
import json
import asyncio
import websockets
import numpy as np
import time

VIDEO_PATH = "test2.mp4"

# 🔥 must match dashboard websocket
WS_URL = "ws://localhost:8000/ws/stream"


async def ws_client():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Could not open video")
        return

    async with websockets.connect(WS_URL) as ws:

        print("Connected to VisionIQ AI engine")

        frame_count = 0
        start_time = time.time()

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Looping video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (640, 480))

            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buffer).decode()

            await ws.send(b64)

            result = json.loads(await ws.recv())

            ann_bytes = base64.b64decode(result["annotated_base64"])
            nparr = np.frombuffer(ann_bytes, np.uint8)
            ann_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            frame_count += 1

            fps = frame_count / (time.time() - start_time)

            print(
                f"Frame {frame_count} | "
                f"People: {result['person_count']} | "
                f"Score: {result['threat_score']} | "
                f"Status: {result['status']} | "
                f"FPS: {fps:.1f}"
            )

            cv2.imshow("Original Video", frame)
            cv2.imshow("VisionIQ AI Output", ann_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()


asyncio.run(ws_client())