# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os, json, ast, psutil
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

# ----------------------------
# Config (Adjust the paths)
# ----------------------------
BASE_DIR = "your_project_dir" # input project directory
DATA_DIR = os.path.join(BASE_DIR, "your_data_dir")  # input data training directory
JSON_HIER_PATH = os.path.join(DATA_DIR, "parent_to_child.json")
TEST_CSV = os.path.join(DATA_DIR, "synthetic_data_test_encoded.csv") # adjust the data path for real-world data inference
RESULT_DIR = os.path.join(BASE_DIR, "Results")
MODEL_BASE_PATH = os.path.join(BASE_DIR, "Models/indobert-lsr")
# adjust the model path for IndoBERT-NLL
MODEL_PATHS = {
    42: [
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L1_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L2_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L3_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L4_seed42.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L5_seed42.pt')
    ],
    112: [
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L1_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L2_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L3_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L4_seed112.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L5_seed112.pt')
    ],
    2025: [
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L1_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L2_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L3_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L4_seed2025.pt'),
        os.path.join(MODEL_BASE_PATH,'indobert_best_lsr_L5_seed2025.pt')
    ]
}


N_OUT = [21, 88, 245, 567, 1789]   # jumlah kelas tiap level
SAVE_DIR = os.path.join(RESULT_DIR, "indobert_lsr_synth")  # adjust the output directory
MAX_LEN = 64
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

def _probs_from_logrow(logits_row):
    # logits_row: 1D tensor berupa log-softmax
    return torch.exp(logits_row)


# ----------------------------
# Utility: load hierarchy (JSON or python dict file)
# ----------------------------
with open(JSON_HIER_PATH, "r") as f:
    content = f.read().strip()

try:
    parent_to_child = json.loads(content)
    print("Loaded hierarchy as JSON.")
except json.JSONDecodeError:
    # fallback if file is python-dict-like
    try:
        if content.startswith("parent_to_child"):
            content = content.split("=", 1)[1].strip()
        parent_to_child = ast.literal_eval(content)
        print("Loaded hierarchy as Python literal.")
    except Exception as e:
        raise RuntimeError("Failed to parse parent_to_child file: " + str(e))

# quick check
print("Hierarchy levels:", len(parent_to_child))
print("example keys:", list(parent_to_child.keys())[:5])


# ----------------------------
# Model class & wrapper
# ----------------------------
class IndoBERTClassifier(nn.Module):
    def __init__(self, output_dim, pretrained_model="indobenchmark/indobert-base-p2"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # we accept token_type_ids even if not used (compatibility)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0]
        out = self.dropout(pooled_output)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)


class ModelWrapper:
    def __init__(self, name, model, tokenizer, device):
        self.name = name
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict_batch(self, texts, batch_size=8):
        """
        Returns:
            preds: list of int predictions (per text)
            logits: list of 1D torch.Tensor (raw logits) on CPU in same order
        """
        self.model.eval()
        all_preds, all_logits = [], []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)   # (batch_size, n_classes) log-prob (we used log_softmax)
                # convert log-probs -> logits_like by exponent then log? we can use outputs (log probs) but consistent:
                # outputs are log-softmax values; for softmax probs we can exp(outputs)
                # keep outputs as cpu tensor per sample
                outputs_cpu = outputs.cpu()
                _, preds = torch.max(outputs_cpu, dim=1)
                all_preds.extend(preds.tolist())
                # store raw outputs_cpu rows as 1D tensors
                for r in outputs_cpu:
                    all_logits.append(r.clone())
        return all_preds, all_logits


# ----------------------------
# Helpers: fallback using cached logits
# ----------------------------
def fallback_topdown_using_logits(pred, conf, current_parent, parent_to_child, logits_row, level):
    """
    pred: predicted idx (int) from cached preds for this level
    conf: initial confidence (float) 
    current_parent: parent idx (int) from previous level (or None)
    parent_to_child: dict 
    logits_row: 1D tensor (log-softmax)
    level: int 
    """
    if current_parent is None:
        # no parent constraint for this sample
        probs = _probs_from_logrow(logits_row)
        return int(pred), float(probs[pred].item())

    # children of current_parent at this mapping level
    valid_children = parent_to_child.get(level, {}).get(current_parent, [])
    if not valid_children:
        probs = _probs_from_logrow(logits_row)
        return int(pred), float(probs[pred].item())

    # compute probs once (from log-softmax)
    probs = _probs_from_logrow(logits_row)

    # if predicted already valid, return it with its prob
    if pred in valid_children:
        return int(pred), float(probs[pred].item())

    # otherwise select valid child with highest log-prob (equivalently highest prob)
    valid_children_in_range = [c for c in valid_children if c < probs.shape[0]]
    if valid_children_in_range:
        # pick child with max log-prob (or max prob)
        best = max(valid_children_in_range, key=lambda c: logits_row[c].item())
        return int(best), float(probs[best].item())

    # fallback default (first valid child) if none in range
    return int(valid_children[0]), float(probs[valid_children[0]].item())



def fallback_bottomup_using_logits(pred_parent, conf, current_child, child_to_parent, logits_row, level):
    """
    Hierarchical logic:
    - child_to_parent[level+1]: mapping from child at level (level+1) -> parent at the current level
    - logits_row: log-softmax (log-probabilities)
    - pred_parent: predicted parent based on cached logits at the current level
    """
    if current_child is None:
        probs = _probs_from_logrow(logits_row)
        return int(pred_parent), float(probs[pred_parent].item())

    # Retrieve valid parents from the next lower level (since a child belongs to a parent)
    valid_parents = child_to_parent.get(level + 1, {}).get(current_child, [])
    if not valid_parents:
        probs = _probs_from_logrow(logits_row)
        return int(pred_parent), float(probs[pred_parent].item())

    probs = _probs_from_logrow(logits_row)

    # If the predicted parent is valid, return it directly along with its confidence
    if pred_parent in valid_parents:
        return int(pred_parent), float(probs[pred_parent].item())

    # If not valid, select the parent with the highest probability among valid candidates
    valid_parents_in_range = [p for p in valid_parents if p < probs.shape[0]]
    if valid_parents_in_range:
        best = max(valid_parents_in_range, key=lambda p: logits_row[p].item())
        return int(best), float(probs[best].item())

    # Fallback: if all valid parents are out of range, return the first available parent
    fallback_parent = valid_parents[0]
    return int(fallback_parent), float(probs[fallback_parent].item())


# ----------------------------
# Efficient forward_all (single pass per model) -> cache preds & logits
# ----------------------------
class EfficientHierarchicalInference:
    def __init__(self, models):
        self.models = models

    def forward_all(self, texts, batch_size=16):
        preds_all, logits_all = [], []
        start_time = time.time()
        # ensure eval mode
        for m in self.models:
            m.model.eval()

        for lvl, model in enumerate(self.models):
            print(f"Inference level {lvl+1} ...")
            preds, logits = model.predict_batch(texts, batch_size=batch_size)
            # logits is list of 1D tensors (log-probs)
            # sanity: convert to torch.stack later; but store as list for memory
            preds_all.append(preds)
            logits_all.append(logits)
        runtime = time.time() - start_time
        print(f"Shared inference finished in {runtime:.2f}s")
        # quick sanity check lengths
        n = len(texts)
        for lvl, l in enumerate(preds_all):
            if len(l) != n or len(logits_all[lvl]) != n:
                raise RuntimeError(f"Length mismatch at level {lvl}: preds {len(l)} logits {len(logits_all[lvl])} expected {n}")
        return preds_all, logits_all, runtime


# ----------------------------
# Strategies using cached logits with original fallback logic
# ----------------------------
class HierarchyAwareStrategies:
    def __init__(self, texts, preds_all, logits_all, parent_to_child):
        self.texts = texts
        self.preds_all = preds_all                # list(level)[sample] -> predicted idx
        self.logits_all = logits_all              # list(level)[sample] -> 1D tensor (log-prob)
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
        preds_td = []
        confs_td = []
        # Level 0 (root) use cached preds & logits (and convert log-probs -> probs)
        lvl0_logits = self.logits_all[0]
        probs0 = [torch.exp(l) for l in lvl0_logits]
        pred0 = [int(torch.argmax(p).item()) for p in probs0]
        conf0 = [float(p[pred0[i]].item()) for i, p in enumerate(probs0)]
        preds_td.append(pred0)
        confs_td.append(conf0)
        current_parent = pred0

        # subsequent levels: apply parent_to_child check + fallback using cached logits
        for lvl in range(1, self.num_levels):
            logits_level = self.logits_all[lvl]
            preds_lvl, confs_lvl = [], []
            for i in range(n):
                parent = current_parent[i]
                # default arg: use raw predicted idx from cached preds
                raw_pred = int(self.preds_all[lvl][i])
                logits_row = logits_level[i]   # 1D tensor log-probs
                # fallback logic but using cached logits

                logits_row = logits_level[i]   # log-softmax 1D tensor
                probs_row = _probs_from_logrow(logits_row)
                init_conf = float(probs_row[raw_pred].item())
                new_pred, new_conf = fallback_topdown_using_logits(raw_pred, init_conf, parent, self.parent_to_child, logits_row, lvl)
                                
                preds_lvl.append(int(new_pred))
                confs_lvl.append(float(new_conf))
            preds_td.append(preds_lvl)
            confs_td.append(confs_lvl)
            current_parent = preds_lvl

        return preds_td, confs_td

    def bottomup(self):
        n = len(self.texts)
        preds_bu = [None]*self.num_levels
        confs_bu = [None]*self.num_levels
        # bottom level:
        lvl = self.num_levels-1
        #print("lvl", lvl)
        logits_level = self.logits_all[lvl]
        probs_lvl = [torch.exp(l) for l in logits_level]
        pred_lvl = [int(torch.argmax(p).item()) for p in probs_lvl]
        conf_lvl = [float(p[pred_lvl[i]].item()) for i, p in enumerate(probs_lvl)]
        preds_bu[lvl] = pred_lvl
        confs_bu[lvl] = conf_lvl
        current_child = pred_lvl
    
        # go upwards
        for lvl in range(self.num_levels - 2, -1, -1):
        #for lvl in range(self.num_levels):
            #print("lvl2", lvl)
            logits_level = self.logits_all[lvl]
            preds_lvl, confs_lvl = [], []
            for i in range(n):
                child = current_child[i]
                #print("child", child)
                #print(i)
                raw_pred = int(self.preds_all[lvl][i])

                logits_row = logits_level[i]
                probs_row = _probs_from_logrow(logits_row)
                init_conf = float(probs_row[raw_pred].item())
                pred_parent, conf_parent = fallback_bottomup_using_logits(raw_pred, init_conf, child, self.child_to_parent, logits_row, lvl)

                preds_lvl.append(int(pred_parent))
                confs_lvl.append(float(conf_parent))
            preds_bu[lvl] = preds_lvl
            confs_bu[lvl] = confs_lvl
            current_child = preds_lvl
        return preds_bu, confs_bu


    def independent(self):
        # simply use cached preds & confs per level (no hierarchy)
        preds_ind = []
        confs_ind = []
        n = len(self.texts)
        for lvl in range(self.num_levels):
            logits_level = self.logits_all[lvl]
            probs = [torch.exp(l) for l in logits_level]
            preds_lvl = [int(torch.argmax(p).item()) for p in probs]
            confs_lvl = [float(p[preds_lvl[i]].item()) for i, p in enumerate(probs)]
            preds_ind.append(preds_lvl)
            confs_ind.append(confs_lvl)
        return preds_ind, confs_ind

    def hybrid_from_td_bu(self, preds_td, confs_td, preds_bu, confs_bu):
        n_levels = self.num_levels
        df = pd.DataFrame({"text": self.texts})
        for lvl in range(n_levels):
            i = lvl + 1
            df[f"level_{i}_td"] = preds_td[lvl]
            df[f"level_{i}_conf_td"] = confs_td[lvl]
            df[f"level_{i}_bu"] = preds_bu[lvl]
            df[f"level_{i}_conf_bu"] = confs_bu[lvl]

        conf_td_cols = [f"level_{i+1}_conf_td" for i in range(n_levels)]
        conf_bu_cols = [f"level_{i+1}_conf_bu" for i in range(n_levels)]
        df["mean_conf_td"] = df[conf_td_cols].mean(axis=1)
        df["mean_conf_bu"] = df[conf_bu_cols].mean(axis=1)
        df["use_td"] = df["mean_conf_td"] > df["mean_conf_bu"]

        for lvl in range(n_levels):
            i = lvl + 1
            df[f"level_{i}_final"] = np.where(df["use_td"], df[f"level_{i}_td"], df[f"level_{i}_bu"])
            df[f"level_{i}_agree"] = df[f"level_{i}_td"] == df[f"level_{i}_bu"]

        preds_hybrid = [df[f"level_{i}_final"].tolist() for i in range(1, n_levels+1)]
        return preds_hybrid, df


# ----------------------------
# Evaluate and save metrics/reports
# ----------------------------
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

    # path accuracy
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


# ----------------------------
# Runner: load models, run shared forward, then strategies, save outputs
# ----------------------------
def run_pipeline(model_wrappers, texts, true_labels, parent_to_child, save_dir=SAVE_DIR):
    # 1) Forward all
    shared = EfficientHierarchicalInference(model_wrappers)
    preds_all, logits_all, runtime_shared = shared.forward_all(texts, batch_size=BATCH_SIZE)

    # 2) Build strategy helper (uses cached preds/logits)
    strat = HierarchyAwareStrategies(texts, preds_all, logits_all, parent_to_child)

    results = {}

    # Top-down (hierarchy-checked using cached logits + fallback)
    preds_td, confs_td = strat.topdown()
    df_td = pd.DataFrame({"text": texts})
    for lvl in range(len(preds_td)):
        df_td[f"level_{lvl+1}"] = preds_td[lvl]
        df_td[f"level_{lvl+1}_conf"] = confs_td[lvl]
    df_td.to_csv(os.path.join(save_dir, "predictions_topdown.csv"), index=False)
    res_td = evaluate_and_save(true_labels, preds_td, texts, os.path.join(save_dir, "reports_topdown"), "topdown", runtime_shared)
    results["topdown"] = res_td

    # Bottom-up
    preds_bu, confs_bu = strat.bottomup()
    df_bu = pd.DataFrame({"text": texts})
    for lvl in range(len(preds_bu)):
        df_bu[f"level_{lvl+1}"] = preds_bu[lvl]
        df_bu[f"level_{lvl+1}_conf"] = confs_bu[lvl]
    df_bu.to_csv(os.path.join(save_dir, "predictions_bottomup.csv"), index=False)
    res_bu = evaluate_and_save(true_labels, preds_bu, texts, os.path.join(save_dir, "reports_bottomup"), "bottomup", runtime_shared)
    results["bottomup"] = res_bu

    # Independent (flat)
    preds_ind, confs_ind = strat.independent()
    df_ind = pd.DataFrame({"text": texts})
    for lvl in range(len(preds_ind)):
        df_ind[f"level_{lvl+1}"] = preds_ind[lvl]
        df_ind[f"level_{lvl+1}_conf"] = confs_ind[lvl]
    df_ind.to_csv(os.path.join(save_dir, "predictions_independent.csv"), index=False)
    res_ind = evaluate_and_save(true_labels, preds_ind, texts, os.path.join(save_dir, "reports_independent"), "independent", runtime_shared)
    results["independent"] = res_ind

    # Hybrid (from TD & BU)
    preds_hybrid, df_hybrid = strat.hybrid_from_td_bu(preds_td, confs_td, preds_bu, confs_bu)
    df_hybrid.to_csv(os.path.join(save_dir, "predictions_hybrid.csv"), index=False)
    res_hybrid = evaluate_and_save(true_labels, preds_hybrid, texts, os.path.join(save_dir, "reports_hybrid"), "hybrid", runtime_shared)
    results["hybrid"] = res_hybrid

    # runtime summary file
    runtime_summary = pd.DataFrame([
        {"strategy": k, "runtime_shared_sec": runtime_shared} for k in results.keys()
    ])
    runtime_summary.to_csv(os.path.join(save_dir, "runtime_summary.csv"), index=False)

    return results


# ----------------------------
# Load models and data, run
# ----------------------------
def build_model_wrappers(model_paths, n_out, device):
    tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
    wrappers = []
    for i, (p, out_dim) in enumerate(zip(model_paths, n_out)):
        model = IndoBERTClassifier(out_dim)
        # load weights (map to device)
        state = torch.load(p, map_location=device)
        model.load_state_dict(state)
        wrapper = ModelWrapper(f"level_{i+1}", model, tokenizer, device)
        wrappers.append(wrapper)
    return wrappers


# ===========================================
# MULTI-SEED INFERENCE + METRIC AGGREGATION
# ===========================================
SEEDS = [42, 112, 2025]
all_results = []
print("Loading test data...")
df_test = pd.read_csv(TEST_CSV)
texts = df_test['text'].tolist()
true_labels = [df_test[f'label_level_{i+1}'].tolist() for i in range(len(N_OUT))]


print(f"Running multi-seed evaluation for {SEEDS} ...")

for seed in SEEDS:
    print(f"\n===== Running seed {seed} =====")

    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # load model for this seed
    model_wrappers = build_model_wrappers(MODEL_PATHS[seed], N_OUT, DEVICE)

    # run pipeline
    save_dir_seed = os.path.join(SAVE_DIR, f"seed_{seed}")
    os.makedirs(save_dir_seed, exist_ok=True)

    results = run_pipeline(model_wrappers, texts, true_labels, parent_to_child, save_dir=save_dir_seed)
    all_results.append(results)

# ============================================================
# Combine metrics from all seeds → compute mean & std per metric
# ============================================================
strategies = ["topdown", "bottomup", "independent", "hybrid"]
summary = {}


for strategy in strategies:
    # collect per-seed DataFrames
    dfs = [r[strategy] for r in all_results]
    merged = pd.concat(dfs, keys=SEEDS, names=["seed", "index"])
    
    # mean and std over all numeric columns
    mean_df = merged.groupby("index").mean(numeric_only=True)
    std_df  = merged.groupby("index").std(numeric_only=True)
    
    # combine side-by-side
    summary_df = mean_df.copy()
    for col in mean_df.columns:
        summary_df[f"{col}_std"] = std_df[col]
    
    summary[strategy] = summary_df
    summary_df.to_csv(os.path.join(SAVE_DIR, f"metrics_summary_{strategy}.csv"))
    print(f"Saved averaged metrics for {strategy}")

# optional: save overall summary table (Path Accuracy comparison)
overall_summary = pd.DataFrame({
    s: {
        "Path_Accuracy_mean": summary[s].loc["Overall", "Path_Accuracy"],
        "Path_Accuracy_std": summary[s].loc["Overall", "Path_Accuracy_std"]
    }
    for s in strategies
}).T
overall_summary.to_csv(os.path.join(SAVE_DIR, "metrics_overall_summary.csv"))
print("\n=== Multi-seed aggregation completed ===")



