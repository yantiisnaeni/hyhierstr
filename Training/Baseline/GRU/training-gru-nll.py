# %%
# ==============================================================
# FastText–GRU Training (multi-seed) - Full Metrics Version
# ==============================================================
import os, time, psutil, random, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# -----------------------------
SEEDS = [42, 112, 2025]
LEVEL = 1                   # adjust the level: 1- 5
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
BASE_DIR = "your_project_dir" # input project directory
MODEL_BASE_PATH = os.path.join(BASE_DIR, "Models/gru-nll")
DATA_DIR = os.path.join(BASE_DIR, "your_data_dir")  # input data training directory
FASTTEXT_PATH = os.path.join(DATA_DIR,"train_vocab_sd_train.pth")
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================
# Dataset Loader
# ==============================================================
def load_dataset(level):
    label_col = f"label_level_{level}"
    train_df = pd.read_csv(f"{DATA_DIR}/synthetic_data_train_encoded.csv", sep=",")
    val_df = pd.read_csv(f"{DATA_DIR}/synthetic_data_val_encoded.csv", sep=",")
    test_df = pd.read_csv(f"{DATA_DIR}/synthetic_data_test_encoded.csv", sep=",")

    cols = ["Text", "Desc", "label_level_1", "label_level_2",
            "label_level_3", "label_level_4", "label_level_5"]
    train_df.columns = cols
    val_df.columns = cols
    test_df.columns = cols

    train_df = train_df[["Desc", label_col]].rename(columns={label_col: "label"})
    val_df = val_df[["Desc", label_col]].rename(columns={label_col: "label"})
    test_df = test_df[["Desc", label_col]].rename(columns={label_col: "label"})

    return train_df, val_df, test_df


# ==============================================================
# Model Definition: FastText–GRU
# ==============================================================
class FastTextGRUClassifier(nn.Module):
    def __init__(self, vocab_vectors, hidden_dim, output_dim, freeze_emb=True, dropout=0.3):
        super().__init__()
        vocab_size, embed_dim = vocab_vectors.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(vocab_vectors)
        self.embedding.weight.requires_grad = not freeze_emb
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, lengths):
        embs = self.embedding(input_ids)
        packed = pack_padded_sequence(embs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        fwd_last = out[range(len(out)), lengths - 1, :self.gru.hidden_size]
        bwd_first = out[:, 0, self.gru.hidden_size:]
        h = torch.cat((fwd_last, bwd_first), dim=1)
        h = self.dropout(h)
        logits = self.fc(h)
        return F.log_softmax(logits, dim=-1)


# ==============================================================
# Tokenization Helper
# ==============================================================
class SimpleVocab:
    def __init__(self, vocab_obj):
        self.itos = vocab_obj.itos
        self.stoi = vocab_obj.stoi
        self.vectors = vocab_obj.vectors

    def encode_batch(self, texts):
        tokenized = [torch.tensor([self.stoi.get(tok, 0) for tok in t.split()], dtype=torch.long)
                     for t in texts]
        lengths = torch.tensor([len(t) for t in tokenized], dtype=torch.long)
        padded = pad_sequence(tokenized, batch_first=True)
        return padded, lengths


# ==============================================================
# Utility Functions
# ==============================================================
def log_resources(epoch, start_time):
    elapsed = time.time() - start_time
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"Epoch {epoch+1:<3} | Time: {elapsed:.2f}s | GPU peak: {gpu_mem:.2f} MB | CPU: {cpu_mem:.2f} MB")


def predict_and_save_results(df, model, vocab, device, batch_size=32, output_csv="pred.csv", metrics_csv="metrics.csv"):
    model.eval()
    texts, true_labels = df["Desc"].tolist(), df["label"].tolist()
    predictions = []

    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            X, lengths = vocab.encode_batch(batch_texts)
            X, lengths = X.to(device), lengths.to(device)
            outputs = model(X, lengths)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    total_time = time.time() - start_time
    avg_time = total_time / len(texts)
    df["predicted_label"] = predictions
    df.to_csv(output_csv, index=False)

    metrics = {
        "Accuracy": accuracy_score(true_labels, predictions),
        "Precision_weighted": precision_score(true_labels, predictions, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(true_labels, predictions, average="weighted", zero_division=0),
        "F1_weighted": f1_score(true_labels, predictions, average="weighted", zero_division=0),
        "Precision_micro": precision_score(true_labels, predictions, average="micro", zero_division=0),
        "Recall_micro": recall_score(true_labels, predictions, average="micro", zero_division=0),
        "F1_micro": f1_score(true_labels, predictions, average="micro", zero_division=0),
        "Precision_macro": precision_score(true_labels, predictions, average="macro", zero_division=0),
        "Recall_macro": recall_score(true_labels, predictions, average="macro", zero_division=0),
        "F1_macro": f1_score(true_labels, predictions, average="macro", zero_division=0),
        "Total Prediction Time (s)": total_time,
        "Avg Time per Record (s)": avg_time
    }

    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    pd.DataFrame(classification_report(true_labels, predictions, output_dict=True)).transpose()\
        .to_csv(metrics_csv.replace(".csv", "_class_report.csv"))
    print(f"Saved predictions and metrics to {metrics_csv}")
    return metrics


# ==============================================================
# Main Training Loop (Multi-Seed)
# ==============================================================
train_df, val_df, test_df = load_dataset(LEVEL)
vocab = SimpleVocab(torch.load(FASTTEXT_PATH))
all_metrics = []

for seed in SEEDS:
    print(f"\n==============================")
    print(f"Training FastText–GRU | Level {LEVEL} | Seed {seed}")
    print(f"==============================")
    set_seed(seed)

    num_classes = len(set(train_df["label"]))
    model = FastTextGRUClassifier(vocab.vectors, hidden_dim=512, output_dim=num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.NLLLoss()
    
    def make_loader(df, shuffle=False):
        X, lengths = vocab.encode_batch(df["Desc"].tolist())
        y = torch.tensor(df["label"].tolist(), dtype=torch.long)
        ds = TensorDataset(X, lengths, y)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df)
    best_val_loss = float("inf")
    model_path = f"{MODEL_BASE_PATH}/gru_best_lsr_L{LEVEL}_seed{seed}.pt"
    epoch_metrics = []

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        # Train
        model.train()
        total_loss, y_true, y_pred = 0, [], []
        for X, lengths, labels in train_loader:
            X, lengths, labels = X.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(X, lengths)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(y_true, y_pred)
        # Validation
        model.eval()
        val_loss, vy_true, vy_pred = 0, [], []
        with torch.no_grad():
            for X, lengths, labels in val_loader:
                X, lengths, labels = X.to(device), lengths.to(device), labels.to(device)
                out = model(X, lengths)
                val_loss += loss_fn(out, labels).item()
                preds = torch.argmax(out, dim=1)
                vy_true.extend(labels.cpu().numpy())
                vy_pred.extend(preds.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = accuracy_score(vy_true, vy_pred)

        log_resources(epoch, start)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

        epoch_record = {
            "Seed": seed, "Epoch": epoch+1,
            "Train_Loss": train_loss, "Val_Loss": val_loss,
            "Train_Acc": train_acc, "Val_Acc": val_acc,
            "Train_F1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "Val_F1_weighted": f1_score(vy_true, vy_pred, average='weighted', zero_division=0)
        }
        epoch_metrics.append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved (epoch {epoch+1})")

    pd.DataFrame(epoch_metrics).to_csv(f"{MODEL_BASE_PATH}/epoch_metrics_L{LEVEL}_seed{seed}.csv", index=False)

    # Testing
    model.load_state_dict(torch.load(model_path))
    metrics = predict_and_save_results(
        test_df.copy(), model, vocab, device,
        output_csv=f"{MODEL_BASE_PATH}/test_pred_L{LEVEL}_seed{seed}.csv",
        metrics_csv=f"{MODEL_BASE_PATH}/test_metrics_L{LEVEL}_seed{seed}.csv"
    )
    metrics["Seed"] = seed
    all_metrics.append(metrics)

# ==============================================================
# Summary
# ==============================================================
summary = pd.DataFrame(all_metrics).describe().loc[["mean", "std"]]
summary.to_csv(f"{MODEL_BASE_PATH}/test_metrics_L{LEVEL}_summary.csv")
print("\n Mean & Std across seeds:")
print(summary)



