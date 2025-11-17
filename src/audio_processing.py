"""
Step 2 — Audio Preprocessing for Real-Time Emotion Coach
---------------------------------------------------------
Uses Librosa to:
- Load each .wav file
- Normalize amplitude
- Trim silence
- Segment into fixed-length chunks (default 5 s)
- Extract MFCCs as simple acoustic features

Outputs:
- processed_features.csv   (metadata + mean MFCCs per clip)
- Optionally saves segmented .wav chunks
"""

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Paths
RAW_DIR = Path("data/raw/ravdess")
METADATA_CSV = Path("data/processed/ravdess_metadata_local.csv")
PROCESSED_AUDIO_DIR = Path("data/processed/segments")
PROCESSED_FEATURES_CSV = Path("data/processed/processed_features.csv")

# Config
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5.0  # seconds
N_MFCC = 40


# ---------------- Core utilities ---------------- #

def load_audio(path, sr=SAMPLE_RATE):
    """Load audio and resample to sr."""
    y, _ = librosa.load(path, sr=sr)
    return y


def normalize(y):
    """Scale waveform to [-1, 1]."""
    if np.max(np.abs(y)) == 0:
        return y
    return y / np.max(np.abs(y))


def trim_silence(y):
    """Trim leading/trailing silence using librosa.effects.trim."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed


def segment_audio(y, sr, segment_duration=SEGMENT_DURATION):
    """Split audio into fixed-length segments."""
    seg_len = int(segment_duration * sr)
    total = len(y)
    segments = []
    for start in range(0, total, seg_len):
        end = min(start + seg_len, total)
        segments.append(y[start:end])
    return segments


def extract_mfcc(y, sr, n_mfcc=N_MFCC):
    """Compute mean-pooled MFCC vector."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


# ---------------- Main pipeline ---------------- #

def process_all_audio(metadata_csv=METADATA_CSV):
    df_meta = pd.read_csv(metadata_csv)
    PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Processing audio"):
        fpath = Path(row["file_path"])
        emotion = row.get("emotion", "unknown")
        actor = row.get("actor", "NA")

        try:
            y = load_audio(fpath)
            y = normalize(trim_silence(y))
            segments = segment_audio(y, SAMPLE_RATE, SEGMENT_DURATION)
            for i, seg in enumerate(segments):
                # skip empty or very short segments
                if len(seg) < 0.5 * SAMPLE_RATE:
                    continue
                mfcc_vec = extract_mfcc(seg, SAMPLE_RATE)
                seg_name = f"{fpath.stem}_seg{i+1}.wav"
                seg_path = PROCESSED_AUDIO_DIR / seg_name
                # Save segment
                sf.write(seg_path, seg, SAMPLE_RATE)
                feature_rows.append({
                    "segment_path": seg_path.as_posix(),
                    "emotion": emotion,
                    "actor": actor,
                    **{f"mfcc_{j+1}": v for j, v in enumerate(mfcc_vec)},
                    "segment_i": i
                })
        except Exception as e:
            print(f" Error processing {fpath.name}: {e}")

    df_feat = pd.DataFrame(feature_rows)
    df_feat.to_csv(PROCESSED_FEATURES_CSV, index=False)
    print(f"Saved processed features to {PROCESSED_FEATURES_CSV}")
    print(f" Saved segments to {PROCESSED_AUDIO_DIR} (count = {len(df_feat)})")
    return df_feat


# ---------------- Run ---------------- #
if __name__ == "__main__":
    df_processed = process_all_audio()
    print(df_processed.head())
