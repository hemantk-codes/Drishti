"""
backend/test_client.py

Quick manual test for /ws/stream, before Phase 5's real frontend exists.
Sends ONE frame (a JPEG file from disk) as base64 text, prints whatever
JSON comes back.

Usage (from the project root, with the server already running):
    python backend/test_client.py
    python backend/test_client.py test_data/some_other_photo.jpg
"""

import asyncio
import base64
import json
import sys

import websockets

DEFAULT_IMAGE = "test_data/sample1room.jpeg"
WS_URL = "ws://localhost:8000/ws/stream"


async def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    with open(image_path, "rb") as f:
        jpg_bytes = f.read()
    b64_str = base64.b64encode(jpg_bytes).decode("utf-8")

    print(f"[test_client] Connecting to {WS_URL} ...")
    async with websockets.connect(WS_URL) as ws:
        print(f"[test_client] Sending {image_path} ({len(jpg_bytes)} bytes) as one frame ...")
        await ws.send(b64_str)
        response = await ws.recv()
        parsed = json.loads(response)
        print("[test_client] Response:")
        print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
