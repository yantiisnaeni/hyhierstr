# %%
# ==============================================================
# IndoBERT–GRU Training (multi-seed, full metrics version)
# ==============================================================
import os, time, psutil, random, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

# -----------------------------
SEEDS = [42, 112, 2025]     # three different seeds
LEVEL = 1                   # adjust the level: 1- 5
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-5
BASE_DIR = "your_project_dir" # input project directory
MODEL_BASE_PATH = os.path.join(BASE_DIR, "Models/indobert-gru-lsr")
DATA_DIR = os.path.join(BASE_DIR, "your_data_dir")  # input data training directory
MODEL_NAME = "indobenchmark/indobert-base-p2"
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================
# Utilities
# ==============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        num_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        return torch.mean(torch.sum(-smoothed * pred, dim=-1))


def log_resources(epoch, start_time):
    elapsed = time.time() - start_time
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"Epoch {epoch+1:<3} | Time: {elapsed:.2f}s | GPU peak: {gpu_mem:.2f} MB | CPU: {cpu_mem:.2f} MB")


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
# Model: IndoBERT–GRU
# ==============================================================
class IndoBERTGRUClassifier(nn.Module):
    def __init__(self, output_dim, pretrained_model="indobenchmark/indobert-base-p2",
                 hidden_dim=512, dropout=0.3, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.gru = nn.GRU(self.bert.config.hidden_size, hidden_dim,
                          num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, L, H]
        gru_out, _ = self.gru(sequence_output)
        # Ambil hidden terakhir dari GRU
        fwd_last = gru_out[:, -1, :self.gru.hidden_size]
        bwd_first = gru_out[:, 0, self.gru.hidden_size:]
        h = torch.cat((fwd_last, bwd_first), dim=1)
        h = self.dropout(h)
        logits = self.fc(h)
        return F.log_softmax(logits, dim=-1)


# ==============================================================
# Evaluation Helper
# ==============================================================
def predict_and_save_results(df, model, tokenizer, device, batch_size=32, output_csv="pred.csv", metrics_csv="metrics.csv"):
    model.eval()
    texts, true_labels = df["Desc"].tolist(), df["label"].tolist()
    predictions = []

    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
            input_ids, attn_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            outputs = model(input_ids, attn_mask)
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
        "Precision_macro": precision_score(true_labels, predictions, average="macro", zero_division=0),
        "Recall_macro": recall_score(true_labels, predictions, average="macro", zero_division=0),
        "F1_macro": f1_score(true_labels, predictions, average="macro", zero_division=0),
        "Total Time (s)": total_time,
        "Avg Time per Record (s)": avg_time
    }

    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    pd.DataFrame(classification_report(true_labels, predictions, output_dict=True)).transpose()\
        .to_csv(metrics_csv.replace(".csv", "_class_report.csv"))
    print(f"Saved predictions and metrics to {metrics_csv}")
    return metrics


# ==============================================================
# Main Training Loop (multi-seed)
# ==============================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_df, val_df, test_df = load_dataset(LEVEL)
all_metrics = []

for seed in SEEDS:
    print(f"\n==============================")
    print(f"Training IndoBERT–GRU | Level {LEVEL} | Seed {seed}")
    print(f"==============================")
    set_seed(seed)

    num_classes = len(set(train_df["label"]))
    model = IndoBERTGRUClassifier(output_dim=num_classes, pretrained_model=MODEL_NAME)
    model = model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = torch.optim.Adam([
        {"params": model.bert.parameters(), "lr": 1e-5},   # IndoBERT fine-tuning
        {"params": model.gru.parameters(), "lr": 1e-3},    # GRU cepat adaptasi
        {"params": model.fc.parameters(), "lr": 1e-3}      # Classifier cepat adaptasi
    ])

    loss_fn = LabelSmoothingLoss(smoothing=0.1)

    def make_loader(df, shuffle=False):
        enc = tokenizer(df["Desc"].tolist(), truncation=True, padding=True, max_length=64, return_tensors="pt")
        y = torch.tensor(df["label"].tolist(), dtype=torch.long)
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], y)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df)
    best_val_loss = float("inf")
    model_path = f"{MODEL_BASE_PATH}/indobert_gru_best_lsr_L{LEVEL}_seed{seed}.pt"
    epoch_records = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        # ---------------- TRAIN ----------------
        model.train()
        train_loss, y_true, y_pred = 0, [], []
        for input_ids, attn_mask, labels in train_loader:
            input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(input_ids, attn_mask)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(y_true, y_pred)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, vy_true, vy_pred = 0, [], []
        with torch.no_grad():
            for input_ids, attn_mask, labels in val_loader:
                input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
                out = model(input_ids, attn_mask)
                val_loss += loss_fn(out, labels).item()
                preds = torch.argmax(out, dim=1)
                vy_true.extend(labels.cpu().numpy())
                vy_pred.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(vy_true, vy_pred)

        log_resources(epoch, start_time)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

        record = {
            "Seed": seed, "Epoch": epoch+1,
            "Train_Loss": train_loss, "Val_Loss": val_loss,
            "Train_Acc": train_acc, "Val_Acc": val_acc,
            "Train_F1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "Val_F1_weighted": f1_score(vy_true, vy_pred, average='weighted', zero_division=0)
        }
        epoch_records.append(record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved (epoch {epoch+1})")

    pd.DataFrame(epoch_records).to_csv(f"{MODEL_BASE_PATH}/epoch_metrics_L{LEVEL}_seed{seed}.csv", index=False)

    # ---------------- TEST ----------------
    model.load_state_dict(torch.load(model_path))
    metrics = predict_and_save_results(
        test_df.copy(), model, tokenizer, device,
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



