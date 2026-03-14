import cv2
import asyncio
import websockets
import base64
import json
import time

VIDEO_PATH = "test3.mp4"
WS_URL     = "ws://localhost:8000/ws/stream"

async def stream():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open {VIDEO_PATH}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 20
    print(f"Video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames @ {fps_video:.0f} FPS")
    print(f"Connecting to {WS_URL} ...")

    async with websockets.connect(WS_URL, max_size=10_000_000) as ws:
        print("Connected! Streaming...\n")

        sent  = 0
        recvd = 0
        t0    = time.time()

        async def sender():
            nonlocal sent
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame = cv2.resize(frame, (640, 480))
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf).decode()

                try:
                    await ws.send(b64)
                    sent += 1
                except Exception as e:
                    print(f"Send error: {e}")
                    break

                await asyncio.sleep(0.05)  # cap at 20 FPS send rate

        async def receiver():
            nonlocal recvd
            async for msg in ws:
                try:
                    data   = json.loads(msg)
                    recvd += 1
                    elapsed = time.time() - t0
                    fps     = recvd / elapsed if elapsed > 0 else 0

                    score   = data.get("threat_score", 0)
                    status  = data.get("status", "?")
                    persons = data.get("person_count", 0)
                    zone    = data.get("zone_triggered", False)

                    icon = "🔴" if status == "CRITICAL" else "🟡" if status == "SUSPICIOUS" else "🟢"
                    print(
                        f"{icon} [{fps:.1f} FPS] Score:{score:3.0f} | "
                        f"{status:<10} | Persons:{persons} | "
                        f"Zone:{'YES' if zone else 'NO '} | "
                        f"Sent:{sent} Recvd:{recvd}"
                    )
                except Exception as e:
                    print(f"Recv error: {e}")

        # Run both concurrently — don't wait for response before sending
        await asyncio.gather(sender(), receiver())

    cap.release()

if __name__ == "__main__":
    try:
        asyncio.run(stream())
    except KeyboardInterrupt:
        print("\nStopped.")
    except ConnectionRefusedError:
        print("ERROR: AI engine not running. Start uvicorn first.")