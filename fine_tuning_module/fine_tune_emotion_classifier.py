import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG =================
SAMPLE_RATE = 16000
MODEL_NAME = "OthmaneJ/distil-wav2vec2"
PROCESSED_FEATURES_CSV = Path("data/processed/processed_features.csv")
BATCH_SIZE = 3
LEARNING_RATE = 2e-5
EPOCHS = 20 # 👈 you can change this anytime

# ================= DATASET =================
class EmotionAudioDataset(Dataset):
    def __init__(self, df, processor, label2id):
        self.df = df
        self.processor = processor
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, _ = librosa.load(row["segment_path"], sr=SAMPLE_RATE)
        inputs = self.processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,return_attention_mask = True)
        label = self.label2id[row["emotion"]]
        return {
            "input_values": inputs.input_values.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ================= MODEL =================
class Wav2Vec2EmotionClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.wav2vec.config.hidden_size, num_labels)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits
    

## padding
def collate_fn(batch):
    """Pads variable-length audio inputs dynamically for Wav2Vec2."""
    input_values = [item["input_values"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    # Pad input_values to the same length
    input_values = torch.nn.utils.rnn.pad_sequence(
        input_values, batch_first=True
    )

    return {"input_values": input_values, "labels": labels}

# ================= TRAINING FUNCTIONS =================
def build_label_map(df):
    emotions = sorted(df["emotion"].unique())
    label2id = {e: i for i, e in enumerate(emotions)}
    id2label = {i: e for e, i in label2id.items()}
    return label2id, id2label

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_values)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

# ================= MAIN RUNNER =================
if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_FEATURES_CSV)
    label2id, id2label = build_label_map(df)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    dataset = EmotionAudioDataset(df, processor, label2id)

    # simple train/val split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_ds = EmotionAudioDataset(train_df, processor, label2id)
    val_ds = EmotionAudioDataset(val_df, processor, label2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2EmotionClassifier(num_labels=len(label2id)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Starting training for {EPOCHS} epochs on {device.upper()}")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

    # Save model checkpoint
    model.wav2vec.save_pretrained("models/wav2vec2_emotion_finetuned")
    processor.save_pretrained("models/wav2vec2_emotion_finetuned")
    print("✅ Model saved as ")
