import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
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

# ===== Config =====
MODEL_PATH = Path("./models/wav2vec2_emotion_finetuned")  # fine-tuned model dir
AUDIO_DIR = Path("data/processed/segments")
OUTPUT_CSV = Path("data/processed/emotion_predictions.csv")
BATCH_SIZE = 4

# ===== 1️⃣ Load model + processor =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True, num_labels=8).to(device)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# ✅ Custom collate function that doesn't stack tensors
def collate_fn(batch):
    file_names = [item[0] for item in batch]
    audio_arrays = [item[1] for item in batch]
    return file_names, audio_arrays

# ===== 2️⃣ Dataset =====
class AudioDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        speech, sr = torchaudio.load(path)
        speech = torch.mean(speech, dim=0)  # convert to mono
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech)
        return path.name, speech.numpy()

audio_files = sorted(AUDIO_DIR.glob("*.wav"))
dataset = AudioDataset(audio_files)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ===== 3️⃣ Inference =====
predictions = []

with torch.no_grad():
    for file_names, audio_arrays in tqdm(loader, desc="Predicting emotions"):
        # Processor handles padding internally
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        print(outputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, preds = torch.max(probs, dim=-1)

        for f, p, c in zip(file_names, preds.cpu(), conf.cpu()):
            predictions.append({
                "file": f,
                "predicted_emotion": id2label[int(p)],
                "confidence": float(c),
            })
        

df = pd.DataFrame(predictions)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved predictions to {OUTPUT_CSV}")