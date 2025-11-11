# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os, json, ast, psutil
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# ----------------------------
# Config (path & hyperparams)
# ----------------------------
JSON_HIER_PATH = "/kaggle/input/parent-to-child/parent_to_child.json"
TEST_CSV = "/kaggle/input/synthesis-data-kbli/sd_test_encoded_subset.csv"

BASE_DIR = "your_project_dir" # input project directory
DATA_DIR = os.path.join(BASE_DIR, "your_data_dir")  # input data training directory
JSON_HIER_PATH = os.path.join(DATA_DIR, "parent_to_child.json")
TEST_CSV = os.path.join(DATA_DIR, "synthetic_data_test_encoded.csv") # adjust the data path for real-world data inference
RESULT_DIR = os.path.join(BASE_DIR, "Results")
MODEL_BASE_PATH = os.path.join(BASE_DIR, "Models/indobert-gru-lsr")
# adjust the model path for IndoBERT-gru-NLL


MODEL_PATHS = {
    42: [
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L1_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L2_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L3_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L4_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L5_seed42.pt')
    ],
    112: [
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L1_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L2_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L3_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L4_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L5_seed112.pt')
    ],
    2025: [
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L1_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L2_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L3_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L4_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_gru_best_lsr_L5_seed2025.pt')
    ]
}

N_OUT = [21, 88, 245, 567, 1789]
SAVE_DIR = os.path.join(RESULT_DIR, "indobert_gru_lsr_synth")  # adjust the output directory
MAX_LEN = 64
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "indobenchmark/indobert-base-p2"

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Utility: JSON Hierarchy Loader
# ----------------------------
with open(JSON_HIER_PATH, "r") as f:
    content = f.read().strip()

try:
    parent_to_child = json.loads(content)
    print("Loaded hierarchy as JSON.")
except json.JSONDecodeError:
    if content.startswith("parent_to_child"):
        content = content.split("=", 1)[1].strip()
    parent_to_child = ast.literal_eval(content)
    print("Loaded hierarchy as Python literal.")

print(f"Hierarchy levels: {len(parent_to_child)}")

# ----------------------------
# Model Definition
# ----------------------------
class IndoBERTGRUClassifier(nn.Module):
    def __init__(self, output_dim, pretrained_model=MODEL_NAME,
                 hidden_dim=512, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.gru = nn.GRU(self.bert.config.hidden_size, hidden_dim,
                          num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = bert_out.last_hidden_state
        gru_out, _ = self.gru(seq_output)
        fwd_last = gru_out[:, -1, :self.gru.hidden_size]
        bwd_first = gru_out[:, 0, self.gru.hidden_size:]
        h = torch.cat((fwd_last, bwd_first), dim=1)
        h = self.dropout(h)
        logits = self.fc(h)
        return F.log_softmax(logits, dim=-1)

# ----------------------------
# Model Wrapper
# ----------------------------
class ModelWrapper:
    def __init__(self, name, model, tokenizer, device):
        self.name = name
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict_batch(self, texts, batch_size=32):
        self.model.eval()
        preds, logits = [], []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                ids, mask = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
                out = self.model(ids, mask)
                preds_batch = torch.argmax(out, dim=1)
                preds.extend(preds_batch.cpu().tolist())
                logits.extend(out.cpu())
        return preds, logits

# ----------------------------
# Probability Helper
# ----------------------------
def _probs_from_logrow(logits_row):
    return torch.exp(logits_row)

# ----------------------------
# Hierarchical Inference
# ----------------------------
class EfficientHierarchicalInference:
    def __init__(self, models):
        self.models = models

    def forward_all(self, texts, batch_size=16):
        preds_all, logits_all = [], []
        start = time.time()
        for m in self.models:
            m.model.eval()
        for lvl, model in enumerate(self.models):
            print(f"Inference Level {lvl+1} ...")
            preds, logits = model.predict_batch(texts, batch_size)
            preds_all.append(preds)
            logits_all.append(logits)
        runtime = time.time() - start
        print(f"✅ Inference Done in {runtime:.2f}s")
        return preds_all, logits_all, runtime

# ----------------------------
# Strategies (TD, BU, etc.)
# ----------------------------
class HierarchyAwareStrategies:
    def __init__(self, texts, preds_all, logits_all, parent_to_child):
        self.texts = texts
        self.preds_all = preds_all
        self.logits_all = logits_all
        self.parent_to_child = parent_to_child
        self.num_levels = len(preds_all)
        self.child_to_parent = self._invert(parent_to_child)

    def _invert(self, parent_to_child):
        inv_all = {}
        for lvl, pcm in parent_to_child.items():
            inv = {}
            for p, children in pcm.items():
                for c in children:
                    inv.setdefault(c, []).append(p)
            inv_all[lvl] = inv
        return inv_all

    def topdown(self):
        n = len(self.texts)
        preds_td, confs_td = [], []
        lvl0_logits = self.logits_all[0]
        probs0 = [torch.exp(l) for l in lvl0_logits]
        pred0 = [int(torch.argmax(p).item()) for p in probs0]
        conf0 = [float(p[pred0[i]].item()) for i, p in enumerate(probs0)]
        preds_td.append(pred0)
        confs_td.append(conf0)
        current_parent = pred0

        for lvl in range(1, self.num_levels):
            logits_lvl = self.logits_all[lvl]
            preds_lvl, confs_lvl = [], []
            for i in range(n):
                parent = current_parent[i]
                logits_row = logits_lvl[i]
                probs = _probs_from_logrow(logits_row)
                pred = int(torch.argmax(probs).item())
                conf = float(probs[pred].item())
                valid_children = self.parent_to_child.get(lvl, {}).get(parent, [])
                if valid_children and pred not in valid_children:
                    valid_children = [c for c in valid_children if c < probs.shape[0]]
                    if valid_children:
                        pred = max(valid_children, key=lambda c: logits_row[c].item())
                        conf = float(probs[pred].item())
                preds_lvl.append(pred)
                confs_lvl.append(conf)
            preds_td.append(preds_lvl)
            confs_td.append(confs_lvl)
            current_parent = preds_lvl
        return preds_td, confs_td

    def bottomup(self):
        n = len(self.texts)
        preds_bu = [None]*self.num_levels
        confs_bu = [None]*self.num_levels
        lvl = self.num_levels - 1
        logits_lvl = self.logits_all[lvl]
        probs_lvl = [torch.exp(l) for l in logits_lvl]
        pred_lvl = [int(torch.argmax(p).item()) for p in probs_lvl]
        conf_lvl = [float(p[pred_lvl[i]].item()) for i, p in enumerate(probs_lvl)]
        preds_bu[lvl] = pred_lvl
        confs_bu[lvl] = conf_lvl
        current_child = pred_lvl

        for lvl in range(self.num_levels - 2, -1, -1):
            logits_lvl = self.logits_all[lvl]
            preds_lvl, confs_lvl = [], []
            for i in range(n):
                logits_row = logits_lvl[i]
                probs = _probs_from_logrow(logits_row)
                pred = int(torch.argmax(probs).item())
                conf = float(probs[pred].item())
                preds_lvl.append(pred)
                confs_lvl.append(conf)
            preds_bu[lvl] = preds_lvl
            confs_bu[lvl] = confs_lvl
        return preds_bu, confs_bu

    def independent(self):
        preds_ind, confs_ind = [], []
        for logits_lvl in self.logits_all:
            probs = [torch.exp(l) for l in logits_lvl]
            preds = [int(torch.argmax(p).item()) for p in probs]
            confs = [float(p[preds[i]].item()) for i, p in enumerate(probs)]
            preds_ind.append(preds)
            confs_ind.append(confs)
        return preds_ind, confs_ind

    def hybrid(self, td, td_conf, bu, bu_conf):
        df = pd.DataFrame({"text": self.texts})
        n_lvl = self.num_levels
        for lvl in range(n_lvl):
            df[f"td_{lvl+1}"] = td[lvl]
            df[f"bu_{lvl+1}"] = bu[lvl]
            df[f"conf_td_{lvl+1}"] = td_conf[lvl]
            df[f"conf_bu_{lvl+1}"] = bu_conf[lvl]
        df["mean_td"] = df[[f"conf_td_{i+1}" for i in range(n_lvl)]].mean(axis=1)
        df["mean_bu"] = df[[f"conf_bu_{i+1}" for i in range(n_lvl)]].mean(axis=1)
        df["use_td"] = df["mean_td"] > df["mean_bu"]
        preds_hybrid = [
            np.where(df["use_td"], df[f"td_{i+1}"], df[f"bu_{i+1}"]).tolist()
            for i in range(n_lvl)
        ]
        return preds_hybrid, df

# ----------------------------
# Evaluation & Reporting
# ----------------------------
def evaluate_and_save(true_labels, preds, texts, out_dir, name, runtime):
    os.makedirs(out_dir, exist_ok=True)
    metrics = {}
    for lvl, y_true in enumerate(true_labels):
        y_pred = preds[lvl]
        acc = accuracy_score(y_true, y_pred)
        p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        metrics[f"L{lvl+1}"] = {"Accuracy": acc, "F1_weighted": f_w, "Precision_weighted": p_w, "Recall_weighted": r_w}
        pd.DataFrame(confusion_matrix(y_true, y_pred)).to_csv(os.path.join(out_dir, f"confusion_{name}_L{lvl+1}.csv"))
        pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()\
            .to_csv(os.path.join(out_dir, f"class_report_{name}_L{lvl+1}.csv"))

    path_acc = np.mean([
        all(true_labels[l][i] == preds[l][i] for l in range(len(preds)))
        for i in range(len(texts))
    ])
    metrics["Overall"] = {"Path_Accuracy": path_acc, "Runtime": runtime}
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(os.path.join(out_dir, f"metrics_{name}.csv"))
    print(f"[{name}] done.")
    return metrics

# ----------------------------
# Pipeline
# ----------------------------
def build_model_wrappers(model_paths, n_out, device):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    wrappers = []
    for i, (p, out_dim) in enumerate(zip(model_paths, n_out)):
        model = IndoBERTGRUClassifier(out_dim)
        model.load_state_dict(torch.load(p, map_location=device))
        wrapper = ModelWrapper(f"lvl_{i+1}", model, tokenizer, device)
        wrappers.append(wrapper)
    return wrappers

# ----------------------------
# Run Inference (Multi-Seed)
# ----------------------------
SEEDS = [42, 112, 2025]
df_test = pd.read_csv(TEST_CSV)
texts = df_test["cleaned_deskripsi"].tolist() if "text" in df_test.columns else df_test["Desc"].tolist()
true_labels = [df_test[f"label_level_{i+1}"].tolist() for i in range(len(N_OUT))]

results_all = {}
for seed in SEEDS:
    print(f"\n===== SEED {seed} =====")
    wrappers = build_model_wrappers(MODEL_PATHS[seed], N_OUT, DEVICE)
    engine = EfficientHierarchicalInference(wrappers)
    preds_all, logits_all, runtime = engine.forward_all(texts)
    strat = HierarchyAwareStrategies(texts, preds_all, logits_all, parent_to_child)

    preds_td, confs_td = strat.topdown()
    preds_bu, confs_bu = strat.bottomup()
    preds_ind, confs_ind = strat.independent()
    preds_hybrid, _ = strat.hybrid(preds_td, confs_td, preds_bu, confs_bu)

    results = {
        "topdown": evaluate_and_save(true_labels, preds_td, texts, f"{SAVE_DIR}/seed_{seed}/td", "topdown", runtime),
        "bottomup": evaluate_and_save(true_labels, preds_bu, texts, f"{SAVE_DIR}/seed_{seed}/bu", "bottomup", runtime),
        "independent": evaluate_and_save(true_labels, preds_ind, texts, f"{SAVE_DIR}/seed_{seed}/ind", "independent", runtime),
        "hybrid": evaluate_and_save(true_labels, preds_hybrid, texts, f"{SAVE_DIR}/seed_{seed}/hyb", "hybrid", runtime),
    }
    results_all[seed] = results

# ============================================================
# Combine Results Across Seeds (fixed version)
# ============================================================

# Daftar strategi yang digunakan
strategies = ["topdown", "bottomup", "independent", "hybrid"]

# Pemetaan nama folder sebenarnya
folder_map = {
    "topdown": "td",
    "bottomup": "bu",
    "independent": "ind",
    "hybrid": "hyb"
}

for s in strategies:
    dfs = []
    folder = folder_map[s]
    for seed in [42, 112, 2025]:  # pastikan sesuai SEEDS kamu
        mpath = f"{SAVE_DIR}/seed_{seed}/{folder}/metrics_{s}.csv"
        if not os.path.exists(mpath):
            print(f"Missing file for seed={seed}, strategy={s}: {mpath}")
            continue
        df = pd.read_csv(mpath)
        df["Seed"] = seed
        dfs.append(df)

    if len(dfs) == 0:
        print(f"No metrics found for strategy {s}, skipping summary.")
        continue

    merged = pd.concat(dfs)
    mean = merged.groupby(merged.index).mean(numeric_only=True)
    std = merged.groupby(merged.index).std(numeric_only=True)
    summary = mean.copy()
    for col in mean.columns:
        if col != "Seed":
            summary[f"{col}_std"] = std[col]

    summary.to_csv(f"{SAVE_DIR}/metrics_summary_{s}.csv", index=True)
    print(f"Saved summary for strategy: {s}")

print("\n All summaries successfully combined.")

print("Hierarchical Inference IndoBERT–GRU completed.")



