
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Root directory of RAVDESS audio
RAVDESS_DIR = Path("data/raw/ravdess")
OUT_CSV = Path("data/processed/ravdess_metadata_local.csv")

# Mapping from RAVDESS emotion code → emotion name
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def parse_ravdess_filename(filename: str) -> dict:

    base = os.path.basename(filename).replace(".wav", "")
    parts = base.split("-")
    if len(parts) != 7:
        return {}
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion_code": parts[2],
        "emotion": EMOTION_MAP.get(parts[2], "unknown"),
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": int(parts[6]),
    }

def list_wav_files(root: Path):
    """Return all WAV file paths recursively."""
    wavs = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(dirpath, f))
    return sorted(wavs)

def build_metadata_dataframe(root: Path):
    """
    Walk through all wav files and build a DataFrame
    with filename, path, and parsed attributes.
    """
    files = list_wav_files(root)
    print(f"Found {len(files)} audio files under {root}")
    rows = []
    for fpath in tqdm(files, desc="Parsing RAVDESS filenames"):
        meta = parse_ravdess_filename(fpath)
        meta["file_path"] = fpath
        meta["filename"] = os.path.basename(fpath)
        rows.append(meta)
    return pd.DataFrame(rows)

def export_metadata_csv(df: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Metadata CSV saved to {out_csv}")

if __name__ == "__main__":
    df = build_metadata_dataframe(RAVDESS_DIR)
    print(df.head())
    print(df["emotion"].value_counts())
    export_metadata_csv(df, OUT_CSV)
