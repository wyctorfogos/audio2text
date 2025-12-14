import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pyannote.audio import Pipeline

# ================================================================
# CONFIG
# ================================================================

AUDIO_FILE = "./data/audio_01.mp3"
WHISPER_MODEL = "openai/whisper-base"

CHUNK_DURATION = 10.0   # segundos
OVERLAP = 1.0           # segundos
SAMPLE_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ================================================================
# MODELOS
# ================================================================

print("\n🔊 Carregando Whisper...")
whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device)

processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
from pyannote.audio import Pipeline

AUDIO_FILE = "./data/audio_01.mp3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# carregar áudio
waveform, sr = torchaudio.load(AUDIO_FILE)

# mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# resample para 16kHz
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

waveform = waveform.to(device)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1"
).to(device)

output = pipeline({
    "waveform": waveform,
    "sample_rate": 16000
})

for segment, _, speaker in output.itertracks(yield_label=True):
    print(
        f"{speaker} speaks between "
        f"t={segment.start:.2f}s and t={segment.end:.2f}s"
    )

    # ============================================================
    # RECORTE CORRETO DO ÁUDIO
    # ============================================================
    start_sample = int(segment.start * SAMPLE_RATE)
    end_sample = int(segment.end * SAMPLE_RATE)

    audio_segment = waveform[0, start_sample:end_sample]

    # ignora segmentos muito curtos
    if audio_segment.numel() < SAMPLE_RATE * 0.2:
        continue

    # ============================================================
    # WHISPER
    # ============================================================
    inputs = processor(
        audio_segment.cpu().numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}

    with torch.no_grad():
        ids = whisper.generate(
            **inputs,
            language="pt",
            task="transcribe",
            max_new_tokens=128
        )

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    print(f"   📝 {text}\n")
