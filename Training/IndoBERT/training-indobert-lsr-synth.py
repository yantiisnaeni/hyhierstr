# %%
# ==============================================================
#  Environment and Configuration Set-up
# ==============================================================
import random
import numpy as np
import pandas as pd
import torch
import os
import time
import psutil
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# -----------------------------
SEEDS = [42, 112, 2025]     # three different seeds
LEVEL = 1                   # adjust the level: 1- 5
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-5
BASE_DIR = "your_project_dir" # input project directory
MODEL_BASE_PATH = os.path.join(BASE_DIR, "Models/indobert-lsr")
DATA_DIR = os.path.join(BASE_DIR, "your_data_dir")  # input data training directory

# -----------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================
# Load Dataset berdasarkan LEVEL
# ==============================================================
def load_dataset(level):
    label_cols = {
        1: "label_level_1",
        2: "label_level_2",
        3: "label_level_3",
        4: "label_level_4",
        5: "label_level_5"
    }
    label_col = label_cols[level]

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

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "val": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    return dataset, test_df


# ==============================================================
# IndoBERTClassifier Model
# ==============================================================
class IndoBERTClassifier(nn.Module):
    def __init__(self, output_dim, pretrained_model="indobenchmark/indobert-base-p2"):
        super(IndoBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = bert_output.last_hidden_state[:, 0]
        out = self.dropout(pooled_output)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.5):
        """
        Constructor for Label Smoothing Loss.
        :param smoothing: Label smoothing factor (e.g., 0.1 for 10% smoothing).
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        :param pred: Log-softmax output from the model (batch_size, num_classes).
        :param target: Ground truth labels (batch_size).
        """
        num_classes = pred.size(-1)
        # One-hot encode the target labels
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        # Apply label smoothing
        smoothed_labels = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        # Compute loss
        return torch.mean(torch.sum(-smoothed_labels * pred, dim=-1))


# ==============================================================
# Logging and Evaluation Function
# ==============================================================
def log_resources(epoch, start_time):
    elapsed = time.time() - start_time
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"Epoch {epoch+1:<3} | Time: {elapsed:.2f}s | GPU peak: {gpu_mem:.2f} MB | CPU: {cpu_mem:.2f} MB")


def predict_and_save_results(df, model, tokenizer, device, batch_size=32, output_csv="predictions.csv", metrics_csv="metrics.csv"):
    model.eval()
    predictions = []
    texts = df["Desc"].tolist()
    true_labels = df["label"].tolist()
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt", max_length=64)
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    total_time = time.time() - start_time
    avg_time_per_record = total_time / len(texts)
    df["predicted_label"] = predictions
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

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
        "Avg Time per Record (s)": avg_time_per_record
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Evaluation metrics saved to {metrics_csv}")

    # Save classification report
    report = classification_report(true_labels, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_csv = metrics_csv.replace(".csv", "_class_report.csv")
    report_df.to_csv(report_csv, index=True)
    print(f"Classification report saved to {report_csv}")

    return metrics


# ==============================================================
# Multi-Seed Training and Evaluation 
# ==============================================================
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2", trust_remote_code=False)
dataset, test_df = load_dataset(LEVEL)
all_metrics = []

for seed in SEEDS:
    print(f"\n==============================")
    print(f"Training IndoBERT on Level {LEVEL} | Seed {seed}")
    print(f"==============================")
    set_seed(seed)

    # Tokenisasi dataset
    encoded_dataset = dataset.map(lambda t: tokenizer(t['Desc'], truncation=True, padding="max_length", max_length=64),
                                  batched=True, load_from_cache_file=False)

    # Dataloader
    train_dataset = TensorDataset(
        torch.tensor(encoded_dataset["train"]["input_ids"]),
        torch.tensor(encoded_dataset["train"]["attention_mask"]),
        torch.tensor(encoded_dataset["train"]["label"])
    )
    val_dataset = TensorDataset(
        torch.tensor(encoded_dataset["val"]["input_ids"]),
        torch.tensor(encoded_dataset["val"]["attention_mask"]),
        torch.tensor(encoded_dataset["val"]["label"])
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    labels_num = len(set(encoded_dataset["train"]["label"]))
    model = IndoBERTClassifier(output_dim=labels_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = LabelSmoothingLoss(smoothing=0.5)

    best_val_loss = float("inf")
    model_path = f"{MODEL_BASE_PATH}/indobert_best_lsr_L{LEVEL}_seed{seed}.pt"

    # ======================
    # Training Loop
    # ======================
    epoch_metrics = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        # ---------------- TRAIN ----------------
        model.train()
        total_loss, all_labels, all_preds = 0, [], []

        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        train_time = time.time() - start_time
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        # --- Train metrics
        train_metrics = {
            "Accuracy": train_acc,
            "Precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "F1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
            "Precision_micro": precision_score(all_labels, all_preds, average="micro", zero_division=0),
            "Recall_micro": recall_score(all_labels, all_preds, average="micro", zero_division=0),
            "F1_micro": f1_score(all_labels, all_preds, average="micro", zero_division=0),
            "Precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "Recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "F1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "Total Time (s)": train_time,
            "Avg Time per Batch (s)": train_time / len(train_loader)
        }

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, val_labels, val_preds = 0, [], []
        val_start = time.time()

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                outputs = model(input_ids, attention_mask)
                val_loss += loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())

        val_time = time.time() - val_start
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)

        val_metrics = {
            "Accuracy": val_acc,
            "Precision_weighted": precision_score(val_labels, val_preds, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(val_labels, val_preds, average="weighted", zero_division=0),
            "F1_weighted": f1_score(val_labels, val_preds, average="weighted", zero_division=0),
            "Precision_micro": precision_score(val_labels, val_preds, average="micro", zero_division=0),
            "Recall_micro": recall_score(val_labels, val_preds, average="micro", zero_division=0),
            "F1_micro": f1_score(val_labels, val_preds, average="micro", zero_division=0),
            "Precision_macro": precision_score(val_labels, val_preds, average="macro", zero_division=0),
            "Recall_macro": recall_score(val_labels, val_preds, average="macro", zero_division=0),
            "F1_macro": f1_score(val_labels, val_preds, average="macro", zero_division=0),
            "Total Time (s)": val_time,
            "Avg Time per Batch (s)": val_time / len(val_loader)
        }

        log_resources(epoch, start_time)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        epoch_record = {
            "Seed": seed,
            "Epoch": epoch + 1,
            "Train_Loss": train_loss,
            **{f"Train_{k}": v for k, v in train_metrics.items()},
            "Val_Loss": avg_val_loss,
            **{f"Val_{k}": v for k, v in val_metrics.items()}
        }
        epoch_metrics.append(epoch_record)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f" Best model saved at epoch {epoch+1} (seed={seed})")

    # Save metrics per epoch
    epoch_df = pd.DataFrame(epoch_metrics)
    epoch_csv = f"{MODEL_BASE_PATH}/epoch_metrics_L{LEVEL}_seed{seed}.csv"
    epoch_df.to_csv(epoch_csv, index=False)
    print(f" Epoch metrics saved to {epoch_csv}")

    # ======================
    # Testing & Evaluation
    # ======================
    model.load_state_dict(torch.load(model_path))
    metrics = predict_and_save_results(
        df=test_df.copy(),
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=BATCH_SIZE,
        output_csv=f"{MODEL_BASE_PATH}/test_pred_L{LEVEL}_seed{seed}.csv",
        metrics_csv=f"{MODEL_BASE_PATH}/test_metrics_L{LEVEL}_seed{seed}.csv"
    )
    metrics["Seed"] = seed
    all_metrics.append(metrics)


# ==============================================================
# Save Mean and Standard Deviation among Seeds
# ==============================================================
metrics_df = pd.DataFrame(all_metrics)
summary = metrics_df.describe().loc[["mean", "std"]]
summary.to_csv(f"{MODEL_BASE_PATH}/test_metrics_L{LEVEL}_summary.csv", index=True)
print("\n Mean and Std deviation from three seeds:")
print(summary)



