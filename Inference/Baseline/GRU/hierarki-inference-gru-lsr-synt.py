# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os, json, ast, psutil
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ----------------------------
# Config (sesuaikan path)
# ----------------------------
BASE_DIR = "your_project_dir" # input project directory
DATA_DIR = os.path.join(BASE_DIR, "your_data_dir")  # input data training directory
JSON_HIER_PATH = os.path.join(DATA_DIR, "parent_to_child.json")
TEST_CSV = os.path.join(DATA_DIR, "synthetic_data_test_encoded.csv") # adjust the data path for real-world data inference
RESULT_DIR = os.path.join(BASE_DIR, "Results")
MODEL_BASE_PATH = os.path.join(BASE_DIR, "Models/gru-lsr")
FASTTEXT_PATH = os.path.join(DATA_DIR,"train_vocab_sd_train.pth")
# adjust the model path for GRU-NLL

MODEL_PATHS = {
    42: [
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L1_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L2_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L3_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L4_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L5_seed42.pt')
    ],
    112: [
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L1_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L2_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L3_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L4_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L5_seed112.pt')
    ],
    2025: [
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L1_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L2_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L3_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L4_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'gru_best_lsr_L5_seed2025.pt')
    ]
}

N_OUT = [21, 88, 245, 567, 1789]
SAVE_DIR = os.path.join(RESULT_DIR, "gru_lsr_synth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================================================
# Helper
# ==============================================================
def _probs_from_logrow(logits_row):
    return torch.exp(logits_row)


# ==============================================================
# Load hierarchy
# ==============================================================
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

print("Hierarchy levels:", len(parent_to_child))
print("Example keys:", list(parent_to_child.keys())[:5])


# ==============================================================
# Model definition (FastText + GRU)
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
# Model Wrapper
# ==============================================================
class ModelWrapper:
    def __init__(self, name, model, vocab, device):
        self.name = name
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device

    def tokenize_text(self, text):
        tokens = text.split()
        token_ids = [self.vocab.stoi.get(t, 0) for t in tokens]
        return torch.tensor(token_ids, dtype=torch.long)

    def predict_batch(self, texts, batch_size=32):
        self.model.eval()
        all_preds, all_logits = [], []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                tokenized = [self.tokenize_text(t) for t in batch]
                lengths = torch.tensor([len(t) for t in tokenized], dtype=torch.long)
                padded = nn.utils.rnn.pad_sequence(tokenized, batch_first=True)
                padded, lengths = padded.to(self.device), lengths.to(self.device)

                outputs = self.model(padded, lengths)
                outputs_cpu = outputs.cpu()
                _, preds = torch.max(outputs_cpu, dim=1)
                all_preds.extend(preds.tolist())
                for r in outputs_cpu:
                    all_logits.append(r.clone())

        return all_preds, all_logits


# ==============================================================
# Fallback functions (identical to IndoBERT version)
# ==============================================================
def fallback_topdown_using_logits(pred, conf, current_parent, parent_to_child, logits_row, level):
    if current_parent is None:
        probs = _probs_from_logrow(logits_row)
        return int(pred), float(probs[pred].item())

    valid_children = parent_to_child.get(level, {}).get(current_parent, [])
    if not valid_children:
        probs = _probs_from_logrow(logits_row)
        return int(pred), float(probs[pred].item())

    probs = _probs_from_logrow(logits_row)
    if pred in valid_children:
        return int(pred), float(probs[pred].item())

    valid_children_in_range = [c for c in valid_children if c < probs.shape[0]]
    if valid_children_in_range:
        best = max(valid_children_in_range, key=lambda c: logits_row[c].item())
        return int(best), float(probs[best].item())

    return int(valid_children[0]), float(probs[valid_children[0]].item())


def fallback_bottomup_using_logits(pred_parent, conf, current_child, child_to_parent, logits_row, level):
    if current_child is None:
        probs = _probs_from_logrow(logits_row)
        return int(pred_parent), float(probs[pred_parent].item())

    valid_parents = child_to_parent.get(level + 1, {}).get(current_child, [])
    if not valid_parents:
        probs = _probs_from_logrow(logits_row)
        return int(pred_parent), float(probs[pred_parent].item())

    probs = _probs_from_logrow(logits_row)
    if pred_parent in valid_parents:
        return int(pred_parent), float(probs[pred_parent].item())

    valid_parents_in_range = [p for p in valid_parents if p < probs.shape[0]]
    if valid_parents_in_range:
        best = max(valid_parents_in_range, key=lambda p: logits_row[p].item())
        return int(best), float(probs[best].item())

    fallback_parent = valid_parents[0]
    return int(fallback_parent), float(probs[fallback_parent].item())


# ==============================================================
# Efficient hierarchical inference (forward_all + strategies)
# ==============================================================
class EfficientHierarchicalInference:
    def __init__(self, models):
        self.models = models

    def forward_all(self, texts, batch_size=32):
        preds_all, logits_all = [], []
        start_time = time.time()
        for lvl, model in enumerate(self.models):
            print(f"Inference level {lvl+1} ...")
            preds, logits = model.predict_batch(texts, batch_size=batch_size)
            preds_all.append(preds)
            logits_all.append(logits)
        runtime = time.time() - start_time
        print(f"Shared inference finished in {runtime:.2f}s")
        return preds_all, logits_all, runtime


# ==============================================================
# Hierarchy-aware strategies
# ==============================================================
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

    def independent(self):
        preds_ind, confs_ind = [], []
        for lvl in range(self.num_levels):
            logits = self.logits_all[lvl]
            probs = [torch.exp(l) for l in logits]
            preds = [int(torch.argmax(p)) for p in probs]
            confs = [float(p[preds[i]].item()) for i, p in enumerate(probs)]
            preds_ind.append(preds)
            confs_ind.append(confs)
        return preds_ind, confs_ind

    def topdown(self):
        n = len(self.texts)
        preds_td, confs_td = [], []
        lvl0_probs = [torch.exp(l) for l in self.logits_all[0]]
        pred0 = [int(torch.argmax(p)) for p in lvl0_probs]
        conf0 = [float(p[pred0[i]].item()) for i, p in enumerate(lvl0_probs)]
        preds_td.append(pred0)
        confs_td.append(conf0)
        current_parent = pred0

        for lvl in range(1, self.num_levels):
            preds_lvl, confs_lvl = [], []
            for i in range(n):
                raw_pred = int(self.preds_all[lvl][i])
                logits_row = self.logits_all[lvl][i]
                probs_row = _probs_from_logrow(logits_row)
                init_conf = float(probs_row[raw_pred].item())
                new_pred, new_conf = fallback_topdown_using_logits(
                    raw_pred, init_conf, current_parent[i], self.parent_to_child, logits_row, lvl)
                preds_lvl.append(new_pred)
                confs_lvl.append(new_conf)
            preds_td.append(preds_lvl)
            confs_td.append(confs_lvl)
            current_parent = preds_lvl
        return preds_td, confs_td

    def bottomup(self):
        n = len(self.texts)
        preds_bu, confs_bu = [None]*self.num_levels, [None]*self.num_levels
        lvl = self.num_levels - 1
        probs_lvl = [torch.exp(l) for l in self.logits_all[lvl]]
        pred_lvl = [int(torch.argmax(p)) for p in probs_lvl]
        conf_lvl = [float(p[pred_lvl[i]].item()) for i, p in enumerate(probs_lvl)]
        preds_bu[lvl], confs_bu[lvl] = pred_lvl, conf_lvl
        current_child = pred_lvl

        for lvl in range(self.num_levels - 2, -1, -1):
            preds_lvl, confs_lvl = [], []
            for i in range(n):
                raw_pred = int(self.preds_all[lvl][i])
                logits_row = self.logits_all[lvl][i]
                probs_row = _probs_from_logrow(logits_row)
                init_conf = float(probs_row[raw_pred].item())
                pred_parent, conf_parent = fallback_bottomup_using_logits(
                    raw_pred, init_conf, current_child[i], self.child_to_parent, logits_row, lvl)
                preds_lvl.append(pred_parent)
                confs_lvl.append(conf_parent)
            preds_bu[lvl], confs_bu[lvl] = preds_lvl, confs_lvl
            current_child = preds_lvl
        return preds_bu, confs_bu

    def hybrid_from_td_bu(self, preds_td, confs_td, preds_bu, confs_bu):
        df = pd.DataFrame({"text": self.texts})
        for lvl in range(self.num_levels):
            i = lvl + 1
            df[f"level_{i}_td"] = preds_td[lvl]
            df[f"level_{i}_conf_td"] = confs_td[lvl]
            df[f"level_{i}_bu"] = preds_bu[lvl]
            df[f"level_{i}_conf_bu"] = confs_bu[lvl]
        df["mean_conf_td"] = df[[f"level_{i+1}_conf_td" for i in range(self.num_levels)]].mean(axis=1)
        df["mean_conf_bu"] = df[[f"level_{i+1}_conf_bu" for i in range(self.num_levels)]].mean(axis=1)
        df["use_td"] = df["mean_conf_td"] > df["mean_conf_bu"]
        preds_hybrid = [
            np.where(df["use_td"], df[f"level_{i}_td"], df[f"level_{i}_bu"]).tolist()
            for i in range(1, self.num_levels + 1)
        ]
        return preds_hybrid, df


# %%
# ==============================================================
# Evaluate and save metrics/reports 
# ==============================================================
def evaluate_and_save(true_labels, pred_labels, texts, out_dir, strategy_name, runtime_sec):
    os.makedirs(out_dir, exist_ok=True)
    num_levels = len(true_labels)
    metrics = {}

    for lvl in range(num_levels):
        y_true = np.array(true_labels[lvl])
        y_pred = np.array(pred_labels[lvl])
        present_labels = np.unique(y_true).tolist()

        acc = accuracy_score(y_true, y_pred)
        p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0, labels=present_labels)
        p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
        p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0, labels=present_labels)

        metrics[f"Level_{lvl+1}"] = {
            "Accuracy": acc,
            "Precision_weighted": p_w, "Recall_weighted": r_w, "F1_weighted": f_w,
            "Precision_micro": p_mic, "Recall_micro": r_mic, "F1_micro": f_mic,
            "Precision_macro": p_mac, "Recall_macro": r_mac, "F1_macro": f_mac,
            "Num_Test_Classes": len(present_labels)
        }

        # confusion matrix & class report (per level)
        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        cm_df = pd.DataFrame(cm, index=[f"T_{l}" for l in present_labels], columns=[f"P_{l}" for l in present_labels])
        cm_df.to_csv(os.path.join(out_dir, f"confusion_{strategy_name}_L{lvl+1}.csv"))

        cr = classification_report(y_true, y_pred, labels=present_labels, output_dict=True, zero_division=0)
        cr_df = pd.DataFrame(cr).transpose()
        cr_df.to_csv(os.path.join(out_dir, f"class_report_{strategy_name}_L{lvl+1}.csv"))

    # Path accuracy
    n_samples = len(texts)
    path_acc = sum(all(true_labels[l][i] == pred_labels[l][i] for l in range(num_levels)) for i in range(n_samples)) / n_samples

    metrics["Overall"] = {
        "Path_Accuracy": path_acc,
        "Runtime_sec": runtime_sec,
        "Runtime_per_sample": runtime_sec / len(texts),
        "Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "Used_Memory_MB": psutil.virtual_memory().used / (1024**2)
    }

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv(os.path.join(out_dir, f"metrics_{strategy_name}.csv"))
    print(f"[{strategy_name}] saved reports to {out_dir}")
    return metrics_df


# ==============================================================
# perform forward_all once, then evaluate each strategy
# ==============================================================
def run_pipeline(model_wrappers, texts, true_labels, parent_to_child, save_dir=SAVE_DIR):
    shared = EfficientHierarchicalInference(model_wrappers)
    preds_all, logits_all, runtime_shared = shared.forward_all(texts, batch_size=BATCH_SIZE)

    strat = HierarchyAwareStrategies(texts, preds_all, logits_all, parent_to_child)
    results = {}

    # Top-Down
    preds_td, confs_td = strat.topdown()
    df_td = pd.DataFrame({"text": texts})
    for lvl in range(len(preds_td)):
        df_td[f"level_{lvl+1}"] = preds_td[lvl]
        df_td[f"level_{lvl+1}_conf"] = confs_td[lvl]
    df_td.to_csv(os.path.join(save_dir, "predictions_topdown.csv"), index=False)
    results["topdown"] = evaluate_and_save(true_labels, preds_td, texts, os.path.join(save_dir, "reports_topdown"), "topdown", runtime_shared)

    # Bottom-Up
    preds_bu, confs_bu = strat.bottomup()
    df_bu = pd.DataFrame({"text": texts})
    for lvl in range(len(preds_bu)):
        df_bu[f"level_{lvl+1}"] = preds_bu[lvl]
        df_bu[f"level_{lvl+1}_conf"] = confs_bu[lvl]
    df_bu.to_csv(os.path.join(save_dir, "predictions_bottomup.csv"), index=False)
    results["bottomup"] = evaluate_and_save(true_labels, preds_bu, texts, os.path.join(save_dir, "reports_bottomup"), "bottomup", runtime_shared)

    # Independent
    preds_ind, confs_ind = strat.independent()
    df_ind = pd.DataFrame({"text": texts})
    for lvl in range(len(preds_ind)):
        df_ind[f"level_{lvl+1}"] = preds_ind[lvl]
        df_ind[f"level_{lvl+1}_conf"] = confs_ind[lvl]
    df_ind.to_csv(os.path.join(save_dir, "predictions_independent.csv"), index=False)
    results["independent"] = evaluate_and_save(true_labels, preds_ind, texts, os.path.join(save_dir, "reports_independent"), "independent", runtime_shared)

    # Hybrid
    preds_hybrid, df_hybrid = strat.hybrid_from_td_bu(preds_td, confs_td, preds_bu, confs_bu)
    df_hybrid.to_csv(os.path.join(save_dir, "predictions_hybrid.csv"), index=False)
    results["hybrid"] = evaluate_and_save(true_labels, preds_hybrid, texts, os.path.join(save_dir, "reports_hybrid"), "hybrid", runtime_shared)

    # runtime summary
    runtime_summary = pd.DataFrame([{"strategy": k, "runtime_shared_sec": runtime_shared} for k in results.keys()])
    runtime_summary.to_csv(os.path.join(save_dir, "runtime_summary.csv"), index=False)

    return results


# ==============================================================
# Build models for each level
# ==============================================================
def build_model_wrappers(model_paths, n_out, vocab, device):
    wrappers = []
    for i, (p, out_dim) in enumerate(zip(model_paths, n_out)):
        model = FastTextGRUClassifier(vocab.vectors, hidden_dim=512, output_dim=out_dim)
        state = torch.load(p, map_location=device)
        model.load_state_dict(state)
        wrapper = ModelWrapper(f"level_{i+1}", model, vocab, device)
        wrappers.append(wrapper)
    return wrappers


# ==============================================================
# MULTI-SEED INFERENCE + METRIC AGGREGATION
# ==============================================================
SEEDS = [42, 112, 2025]
all_results = []

print("Loading vocab...")
vocab = torch.load(FASTTEXT_PATH)

print("Loading test data...")
df_test = pd.read_csv(TEST_CSV)
texts = df_test["text"].tolist()
true_labels = [df_test[f"label_level_{i+1}"].tolist() for i in range(len(N_OUT))]

print(f"Running multi-seed evaluation for {SEEDS} ...")

for seed in SEEDS:
    print(f"\n===== Running seed {seed} =====")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model for this seed
    model_wrappers = build_model_wrappers(MODEL_PATHS[seed], N_OUT, vocab, DEVICE)

    # Run pipeline
    save_dir_seed = os.path.join(SAVE_DIR, f"seed_{seed}")
    os.makedirs(save_dir_seed, exist_ok=True)

    results = run_pipeline(model_wrappers, texts, true_labels, parent_to_child, save_dir=save_dir_seed)
    all_results.append(results)


# ==============================================================
# Combine metrics from all seeds → compute mean & std per metric
# ==============================================================
strategies = ["topdown", "bottomup", "independent", "hybrid"]
summary = {}

for strategy in strategies:
    dfs = [r[strategy] for r in all_results]
    merged = pd.concat(dfs, keys=SEEDS, names=["seed", "index"])
    mean_df = merged.groupby("index").mean(numeric_only=True)
    std_df = merged.groupby("index").std(numeric_only=True)
    summary_df = mean_df.copy()
    for col in mean_df.columns:
        summary_df[f"{col}_std"] = std_df[col]
    summary[strategy] = summary_df
    summary_df.to_csv(os.path.join(SAVE_DIR, f"metrics_summary_{strategy}.csv"))
    print(f"Saved averaged metrics for {strategy}")

# overall summary (path accuracy)
overall_summary = pd.DataFrame({
    s: {
        "Path_Accuracy_mean": summary[s].loc["Overall", "Path_Accuracy"],
        "Path_Accuracy_std": summary[s].loc["Overall", "Path_Accuracy_std"]
    }
    for s in strategies
}).T
overall_summary.to_csv(os.path.join(SAVE_DIR, "metrics_overall_summary.csv"))

print("\n=== Multi-seed aggregation completed ===")
print("All results saved under:", SAVE_DIR)



