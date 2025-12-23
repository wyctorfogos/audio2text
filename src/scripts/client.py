import asyncio
import json
import librosa
import numpy as np
import websockets
import uuid
import time

WS_URL = "ws://localhost:8010/ws/stream"
AUDIO_PATH = "./data/audio_01.mp3"

FRAME_MS = 20
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = int(SAMPLE_RATE * (FRAME_MS / 1000.0))

async def main():
    session_id = str(uuid.uuid4())

    async with websockets.connect(WS_URL, max_size=None) as ws:
        # servidor diz "ready"
        print(await ws.recv())

        # inicia sessão
        await ws.send(json.dumps({
            "event": "start",
            "session_id": session_id
        }))
        print(await ws.recv())

        # --------------------------------------------------
        # Carrega áudio (mono, 16kHz)
        # --------------------------------------------------
        y, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)

        # normaliza e converte float32 [-1,1] -> PCM16
        y = np.clip(y, -1.0, 1.0)
        pcm16 = (y * 32767).astype(np.int16)

        total_samples = len(pcm16)
        idx = 0

        print(f"🎧 Enviando áudio | samples={total_samples}")

        # --------------------------------------------------
        # Envia frames de áudio
        # --------------------------------------------------
        while idx < total_samples:
            frame = pcm16[idx:idx + SAMPLES_PER_FRAME]
            if len(frame) == 0:
                break

            await ws.send(frame.tobytes())

            idx += SAMPLES_PER_FRAME
            await asyncio.sleep(FRAME_MS / 1000.0)

        # --------------------------------------------------
        # Finaliza
        # --------------------------------------------------
        await ws.send(json.dumps({"event": "end"}))

        # --------------------------------------------------
        # Recebe mensagens restantes
        # --------------------------------------------------
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                print("⬅️", msg)
        except asyncio.TimeoutError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
