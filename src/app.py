#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diarização (pyannote) + Transcrição (Whisper HF) por segmento.
Saída: JSON Lines (um registro por linha) para ser robusto com append.

Requisitos (exemplos):
  pip install torch torchaudio transformers pyannote.audio

Observações:
- O modelo "pyannote/speaker-diarization-3.1" normalmente exige token da Hugging Face
  (via env var HUGGINGFACE_TOKEN ou login). Se faltar, vai falhar no download.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pyannote.audio import Pipeline


# ================================================================
# CONFIG
# ================================================================
AUDIO_FILE = "./data/audio_01.mp3"
WHISPER_MODEL = "openai/whisper-base"

TARGET_SAMPLE_RATE = 16000
MIN_SEGMENT_SECONDS = 0.20  # ignora segmentos muito curtos (200ms)

# Saída (JSON Lines)
RUN_ID = str(uuid.uuid4())
OUT_PATH = f"./data/diarization_{RUN_ID}.jsonl"

# Token (se necessário)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


# ================================================================
# HELPERS
# ================================================================
def load_audio_mono_16k(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Carrega áudio com torchaudio, converte para mono e reamostra para target_sr.
    Retorna tensor 1D (T,) em float32 no CPU.
    """
    waveform, sr = torchaudio.load(path)  # (C, T)

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # (1, T) -> (T,)
    waveform = waveform.squeeze(0).contiguous()

    # float32 para garantir compatibilidade na diarização
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

    return waveform


def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ================================================================
# MAIN
# ================================================================
def main() -> None:
    # device e dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f" Device: {device} | dtype: {dtype}")
    print(f" Áudio: {AUDIO_FILE}")
    print(f" Saída: {OUT_PATH}")

    # -----------------------------
    # Carrega Whisper
    # -----------------------------
    print("\n Carregando Whisper...")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    whisper.eval()

    processor = AutoProcessor.from_pretrained(WHISPER_MODEL)

    # forçar idioma/tarefa (forma mais robusta no HF)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="pt", task="transcribe")

    # -----------------------------
    # Carrega Diarização
    # -----------------------------
    print("\n Carregando pipeline de diarização (pyannote)...")
    if HF_TOKEN:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    # pyannote aceita device cpu/cuda
    pipeline.to(device)

    # -----------------------------
    # Carrega áudio
    # -----------------------------
    waveform = load_audio_mono_16k(AUDIO_FILE, target_sr=TARGET_SAMPLE_RATE)
    num_samples = waveform.numel()
    duration_s = num_samples / TARGET_SAMPLE_RATE
    print(f"\n Duração: {duration_s:.2f}s | samples: {num_samples}")

    # pyannote espera waveform (C, T). Vamos (1, T)
    diar_waveform = waveform.unsqueeze(0).to(device)

    # -----------------------------
    # Executa diarização
    # -----------------------------
    print("\n🧩 Rodando diarização...")
    diarization = pipeline({"waveform": diar_waveform, "sample_rate": TARGET_SAMPLE_RATE})

    # header da execução (primeira linha do jsonl, útil pra rastrear)
    write_jsonl(
        OUT_PATH,
        {
            "type": "run_meta",
            "run_id": RUN_ID,
            "audio_file": AUDIO_FILE,
            "whisper_model": WHISPER_MODEL,
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "sample_rate": TARGET_SAMPLE_RATE,
            "created_at_utc": datetime.utcnow().isoformat(),
            "device": str(device),
        },
    )

    # -----------------------------
    # Itera segmentos e transcreve
    # -----------------------------
    print("\n Transcrevendo por speaker...")
    seg_idx = 0
    min_samples = int(MIN_SEGMENT_SECONDS * TARGET_SAMPLE_RATE)

    # Vamos usar waveform no CPU para recortar rápido e evitar VRAM extra.
    waveform_cpu = waveform  # já está no CPU

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_s = float(segment.start)
        end_s = float(segment.end)

        start_sample = max(0, int(start_s * TARGET_SAMPLE_RATE))
        end_sample = min(num_samples, int(end_s * TARGET_SAMPLE_RATE))

        audio_segment = waveform_cpu[start_sample:end_sample]

        if audio_segment.numel() < min_samples:
            continue

        # processor aceita np.ndarray 1D
        inputs = processor(
            audio_segment.numpy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )

        # move inputs pro device
        inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = whisper.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=128,
            )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        record = {
            "type": "segment",
            "run_id": RUN_ID,
            "segment_index": seg_idx,
            "speaker": speaker,
            "start": round(start_s, 3),
            "end": round(end_s, 3),
            "start_sample": start_sample,
            "end_sample": end_sample,
            "text": text,
            "language": "pt",
            "created_at_utc": datetime.utcnow().isoformat(),
        }

        print(f"[{seg_idx:04d}] {speaker}  t={start_s:.2f}-{end_s:.2f}s  📝 {text}")
        write_jsonl(OUT_PATH, record)

        seg_idx += 1

    print(f"\n Finalizado. Segmentos salvos: {seg_idx}")
    print(f" Arquivo: {OUT_PATH}")


if __name__ == "__main__":
    main()
