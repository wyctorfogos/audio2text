# api_stream.py
import os
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import uvicorn
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pyannote.audio import Pipeline

# =========================
# CONFIG
# =========================
WHISPER_MODEL = "openai/whisper-small"
DIAR_MODEL = "pyannote/speaker-diarization-3.1"

SAMPLE_RATE = 16000
PCM_DTYPE = np.int16

# Janela deslizante (trade-off latência vs qualidade de diarização)
WINDOW_SECONDS = 30.0          # quanto de contexto mantém
PROCESS_EVERY_SECONDS = 2.0    # roda diarização+ASR a cada X segundos
MIN_SEGMENT_SECONDS = 0.25     # ignora segmentos muito curtos
MAX_NEW_TOKENS = 128

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

app = FastAPI()

# =========================
# MODELS (carregados 1x)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

print(f"🧠 Device: {device} | dtype: {dtype}")

print("🔊 Carregando Whisper...")
whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device).eval()

processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="pt", task="transcribe")

print("🧑‍🤝‍🧑 Carregando pyannote diarization...")
if HF_TOKEN:
    diar_pipeline = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=HF_TOKEN)
else:
    diar_pipeline = Pipeline.from_pretrained(DIAR_MODEL)
diar_pipeline.to(device)

# =========================
# SESSION STATE
# =========================
@dataclass
class StreamState:
    session_id: str
    # buffer PCM16 bytes (mono, 16k)
    audio_bytes: bytearray = field(default_factory=bytearray)
    # tempo (em segundos) do áudio total acumulado (aprox)
    total_samples: int = 0
    # último tempo do áudio que já “emitimos” para o cliente
    last_emitted_t: float = 0.0
    # controle
    running: bool = True
    last_process_time: float = 0.0


def pcm16_bytes_to_float32_tensor(pcm_bytes: bytes) -> torch.Tensor:
    """PCM16 mono -> torch float32 (T,) no CPU."""
    x = np.frombuffer(pcm_bytes, dtype=PCM_DTYPE).astype(np.float32) / 32768.0
    return torch.from_numpy(x)  # CPU float32


def seconds_from_samples(n: int) -> float:
    return n / float(SAMPLE_RATE)


async def send_json(ws: WebSocket, obj: Dict[str, Any]) -> None:
    await ws.send_text(json.dumps(obj, ensure_ascii=False))


def run_diarization_and_asr(window_audio: torch.Tensor,
                           window_offset_t: float,
                           last_emitted_t: float) -> List[Dict[str, Any]]:
    """
    Executa diarização+ASR no áudio da janela.
    Retorna segmentos novos (start/end/text/speaker) cujo start >= last_emitted_t.
    window_offset_t = tempo (s) do início da janela no timeline global.
    """
    # pyannote espera (1, T) e no mesmo device
    wav_1t = window_audio.unsqueeze(0).to(device)

    diar = diar_pipeline({"waveform": wav_1t, "sample_rate": SAMPLE_RATE})

    results: List[Dict[str, Any]] = []
    min_samples = int(MIN_SEGMENT_SECONDS * SAMPLE_RATE)

    # Vamos recortar do window_audio no CPU para o processor
    window_cpu = window_audio.cpu()

    for segment, _, speaker in diar.itertracks(yield_label=True):
        start_t = float(segment.start) + window_offset_t
        end_t = float(segment.end) + window_offset_t

        # só emite “novidades”
        if start_t < last_emitted_t:
            continue

        # recorte no referencial da janela
        start_samp = int(segment.start * SAMPLE_RATE)
        end_samp = int(segment.end * SAMPLE_RATE)
        start_samp = max(0, start_samp)
        end_samp = min(window_cpu.numel(), end_samp)

        clip = window_cpu[start_samp:end_samp]
        if clip.numel() < min_samples:
            continue

        # Whisper
        inputs = processor(
            clip.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}

        with torch.no_grad():
            ids = whisper.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        results.append({
            "type": "segment",
            "speaker": speaker,
            "start": round(start_t, 3),
            "end": round(end_t, 3),
            "text": text,
        })

    # ordena por tempo
    results.sort(key=lambda r: r["start"])
    return results


async def processing_loop(ws: WebSocket, st: StreamState) -> None:
    """
    Loop que periodicamente processa a janela e envia resultados.
    """
    await send_json(ws, {
        "type": "info",
        "session_id": st.session_id,
        "status": "processing_started",
        "device": str(device),
        "window_seconds": WINDOW_SECONDS,
        "process_every_seconds": PROCESS_EVERY_SECONDS
    })

    while st.running:
        await asyncio.sleep(0.05)

        now = time.time()
        if (now - st.last_process_time) < PROCESS_EVERY_SECONDS:
            continue

        st.last_process_time = now

        # calcula janela atual
        total_t = seconds_from_samples(st.total_samples)
        window_start_t = max(0.0, total_t - WINDOW_SECONDS)

        # quantos samples na janela?
        window_start_sample = int(window_start_t * SAMPLE_RATE)
        window_bytes_start = window_start_sample * 2  # int16 => 2 bytes

        # recorta bytes do buffer
        if window_bytes_start >= len(st.audio_bytes):
            continue

        window_pcm = bytes(st.audio_bytes[window_bytes_start:])
        if len(window_pcm) < 2 * int(1.0 * SAMPLE_RATE):  # pelo menos 1s
            continue

        window_audio = pcm16_bytes_to_float32_tensor(window_pcm)  # CPU float32

        try:
            segs = run_diarization_and_asr(
                window_audio=window_audio,
                window_offset_t=window_start_t,
                last_emitted_t=st.last_emitted_t
            )
        except Exception as e:
            # erro do modelo não pode derrubar websocket inteiro
            return await send_json(ws, {"type": "error", "message": str(e)})

        # envia segmentos e avança last_emitted_t
        for s in segs:
            await send_json(ws, s)
            st.last_emitted_t = max(st.last_emitted_t, float(s["end"]))

        # heartbeat opcional
        await send_json(ws, {
            "type": "heartbeat",
            "t": round(total_t, 2),
            "emitted_until": round(st.last_emitted_t, 2),
        })


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    state: Optional[StreamState] = None
    processor_task: Optional[asyncio.Task] = None

    try:
        await send_json(ws, {"type": "ready", "message": "Send {event:start} and then PCM16 frames."})

        while True:
            msg = await ws.receive()

            # cliente desconectou
            if msg["type"] == "websocket.disconnect":
                break

            # controle JSON
            if "text" in msg and msg["text"]:
                obj = json.loads(msg["text"])

                if obj.get("event") == "start":
                    sid = obj.get("session_id") or "session"
                    state = StreamState(session_id=sid)
                    processor_task = asyncio.create_task(processing_loop(ws, state))
                    await send_json(ws, {"type": "ok", "status": "session_ready", "session_id": sid})
                    continue

                if obj.get("event") == "end":
                    if state:
                        state.running = False
                    if processor_task:
                        processor_task.cancel()
                    await send_json(ws, {"type": "ok", "status": "session_ended"})
                    break

            # áudio binário
            if "bytes" in msg and msg["bytes"] is not None:
                if state is None:
                    await send_json(ws, {"type": "error", "message": "Send {event:start} first."})
                    continue

                chunk = msg["bytes"]
                # acumula
                state.audio_bytes.extend(chunk)

                # atualiza contagem de samples (PCM16 => 2 bytes por sample)
                state.total_samples = len(state.audio_bytes) // 2

    except WebSocketDisconnect:
        pass
    finally:
        if state:
            state.running = False
        if processor_task:
            processor_task.cancel()


if __name__=="__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8010)