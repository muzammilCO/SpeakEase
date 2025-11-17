from faster_whisper import WhisperModel
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# ===== Config =====
MODEL_SIZE = "small"        # 'tiny', 'base', 'small', 'medium', 'large'
AUDIO_DIR = Path("data/processed/segments")
OUTPUT_CSV = Path("data/processed/transcriptions.csv")
OUTPUT_JSON = Path("data/processed/transcriptions_detailed.json")

# ===== 1️⃣ Load model =====
print(f"🔊 Loading Whisper model: {MODEL_SIZE} ...")
model = WhisperModel(MODEL_SIZE)   



# ===== 2️⃣ Collect audio files =====
audio_files = sorted(AUDIO_DIR.glob("*.wav"))
print(f"Found {len(audio_files)} audio clips to transcribe.")

# ===== 3️⃣ Transcription loop =====
records = []
details = []

for wav_path in tqdm(audio_files, desc="Transcribing"):
    try:
        # Run Whisper
        segments, info = model.transcribe(str(wav_path))
        segments = [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in segments
        ]
        text = " ".join([seg["text"] for seg in segments])



        # Duration info is returned separately
        duration = info.duration

        # Save summary
        records.append({
            "file": wav_path.name,
            "text": text,
            "duration": duration,
        })

        # Save detailed segment-level transcript
        details.append({
            "file": wav_path.name,
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip()
                }
                for seg in segments
            ],
        })

    except Exception as e:
        print(f"⚠️ Failed on {wav_path.name}: {e}")

# ===== 4️⃣ Save outputs =====
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
with open(OUTPUT_JSON, "w") as f:
    json.dump(details, f, indent=2)

print(f"✅ Saved {len(df)} transcriptions to {OUTPUT_CSV}")
print(f"✅ Detailed JSON saved to {OUTPUT_JSON}")
