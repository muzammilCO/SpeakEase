import torch
import torchaudio
import whisper
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from pathlib import Path
from tqdm import tqdm

# ======================
# CONFIGURATION
# ======================
WHISPER_MODEL_NAME = "small"  # can use "base" or "tiny" if limited GPU
EMOTION_MODEL_PATH = Path("./models/wav2vec2_emotion_finetuned")  # your Step 3 output
AUDIO_FILE = Path("data/raw/Peter Dinklage  Powerful speech on failure  explore  speech  powerful.wav")  # your call recording file
OUTPUT_CSV = Path("data/processed/final_inference_sample_call.csv")

id2label = {
0 : 'angry',
 1 : 'calm',
 2 : 'disgust',
 3 : 'fearful',
 4 : 'happy',
 5 : 'neutral',
 6 : 'sad',
 7 : 'surprised'
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 1️⃣ LOAD MODELS
# ======================
print("🔹 Loading models...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)

emo_processor = Wav2Vec2Processor.from_pretrained(EMOTION_MODEL_PATH, local_files_only=True)
emo_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH, local_files_only=True,num_labels=8).to(device)
emo_model.eval()
print("✅ Models loaded successfully.")

# ======================
# 2️⃣ TRANSCRIBE AUDIO
# ======================
print("🎧 Running Whisper transcription...")
result = whisper_model.transcribe(str(AUDIO_FILE), word_timestamps=True)
segments = result.get("segments", [])
print(segments)
print(f"✅ Transcription complete — {len(segments)} segments detected.")

# ======================
# 3️⃣ EMOTION ANALYSIS PER SEGMENT
# ======================
print("🧠 Running emotion inference...")
records = []

# Pre-load entire waveform once
waveform, sr = torchaudio.load(AUDIO_FILE)

for seg in tqdm(segments, desc="Analyzing emotions"):
    start, end = seg["start"], seg["end"]
    text = seg["text"].strip()

    # Extract segment waveform
    start_sample, end_sample = int(start * sr), int(end * sr)
    segment_wave = waveform[:, start_sample:end_sample]

    # Convert stereo → mono + resample
    segment_wave = torch.mean(segment_wave, dim=0)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    segment_wave = resampler(segment_wave)

    # Prepare inputs
    inputs = emo_processor(segment_wave.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict emotion
    with torch.no_grad():
        outputs = emo_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)

    records.append({
        "start_time": start,
        "end_time": end,
        "text": text,
        "predicted_emotion": id2label[int(pred)],
        "confidence": float(conf),
    })

# ======================
# 4️⃣ SAVE FINAL RESULTS
# ======================
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Combined inference saved to {OUTPUT_CSV}")

# ======================
# 5️⃣ OPTIONAL PREVIEW
# ======================
print(df.head())
