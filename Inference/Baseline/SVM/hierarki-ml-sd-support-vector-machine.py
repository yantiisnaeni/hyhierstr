# %%
import os
import time
import json
import ast
import psutil
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# ----------------------------
# CONFIG - adjust paths here
# ----------------------------
BASE_DIR = "your_project_dir" # input project directory
DATA_DIR = os.path.join(BASE_DIR, "your_data_dir")  # input data training directory
MODEL_BASE_PATH = os.path.join(BASE_DIR, "Models/machine_learning")
RESULT_DIR = os.path.join(BASE_DIR, "Results")

JSON_HIER_PATH = os.path.join(DATA_DIR, "parent_to_child.json")
TEST_CSV = os.path.join(DATA_DIR, "synthetic_data_test_encoded.csv") 

# Template paths for per-level models/vectorizers (adjust names/dirs if needed)
MODEL_PATH_TEMPLATE = os.path.join(MODEL_BASE_PATH, "SVM_model_level_{lvl}.joblib")
VECTORIZER_PATH_TEMPLATE = os.path.join(MODEL_BASE_PATH,"tfidf_vectorizer.joblib")

SAVE_DIR = os.path.join(RESULT_DIR,"hier_out_svm")
os.makedirs(SAVE_DIR, exist_ok=True)

LEVELS = ["1", "2", "3", "4", "5"]
DEVICE = "CPU"

# Batch size for TF-IDF transform / prob prediction (tune if memory limited)
BATCH_SIZE = 2048

# ----------------------------
# Load hierarchy (JSON or Python literal)
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
print("Hierarchy levels:", len(parent_to_child))

# Build child_to_parent map (for bottom-up)
child_to_parent = {}
for parent_level_idx, mapping in parent_to_child.items():
    child_level_idx = parent_level_idx + 1
    inv = {}
    for parent_label, child_labels in mapping.items():
        for c in child_labels:
            # If multiple parents exist for a child, keep the first (KBLI usually single parent)
            inv[c] = parent_label
    child_to_parent[child_level_idx] = inv
print("Built child_to_parent mapping.")

# ----------------------------
# Load test data & models
# ----------------------------
print("Loading test data...")
df_test = pd.read_csv(TEST_CSV)
texts = df_test["cleaned_deskripsi"].astype(str).tolist()
n_samples = len(texts)
true_labels = [df_test[f"label_level_{lvl}"].tolist() for lvl in LEVELS]
print(f"Loaded {n_samples} samples.")

print("Loading models and vectorizers...")
models = {lvl: joblib.load(MODEL_PATH_TEMPLATE.format(lvl=lvl)) for lvl in LEVELS}
vectorizers = {lvl: joblib.load(VECTORIZER_PATH_TEMPLATE.format(lvl=lvl)) for lvl in LEVELS}
print("Models & vectorizers loaded.")

# Precompute class label -> index maps for each model for quick lookup
class_to_index = {}
for lvl in LEVELS:
    classes = list(models[lvl].classes_)
    class_to_index[lvl] = {c: i for i, c in enumerate(classes)}

# ----------------------------
# Helper: batched transform & predict_proba
# ----------------------------
def batched_predict_proba(texts, model, vectorizer, batch_size=BATCH_SIZE):
    """
    Returns:
      probs: np.ndarray shape (n_samples, n_classes) (stacked)
      classes: array-like model.classes_
    Batched to avoid huge memory use if texts large.
    """
    n = len(texts)
    out_probs = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        X = vectorizer.transform(batch_texts)
        probs = model.predict_proba(X)  # shape (batch, n_classes)
        out_probs.append(probs)
    return np.vstack(out_probs), list(model.classes_)


# ----------------------------
# 1) Independent (flat) predictions - batched per level
# ----------------------------
def predict_independent(texts, models, vectorizers, batch_size=BATCH_SIZE):
    preds_all = []
    probs_all = []  # list of arrays
    for lvl in LEVELS:
        probs, classes = batched_predict_proba(texts, models[lvl], vectorizers[lvl], batch_size=batch_size)
        preds_idx = np.argmax(probs, axis=1)
        preds = [classes[i] for i in preds_idx]
        preds_all.append(preds)
        probs_all.append(probs)  
    return preds_all, probs_all  # preds_all: list(level)[sample], probs_all: list(level) -> np.array (n, n_classes)


# ----------------------------
# 2) Top-down (uses cached probs, fallback selects valid child with highest prob)
# ----------------------------
def predict_topdown_from_probs(probs_all, models, parent_to_child):
    """
    probs_all: list(level) -> np.array shape (n_samples, n_classes_level)
    models: dict lvl->model (for classes ordering)
    """
    n_levels = len(probs_all)
    n = probs_all[0].shape[0]
    preds_td = [[] for _ in range(n_levels)]
    confs_td = [[] for _ in range(n_levels)]

    # level 0 (L1) independent
    probs = probs_all[0]
    idx = np.argmax(probs, axis=1)
    classes = list(models[LEVELS[0]].classes_)
    preds_lvl = [classes[i] for i in idx]
    confs_lvl = probs[np.arange(n), idx].tolist()
    preds_td[0] = preds_lvl
    confs_td[0] = confs_lvl
    current_parent = preds_lvl

    # subsequent levels
    for lvl_idx in range(1, n_levels):
        probs = probs_all[lvl_idx]
        model = models[LEVELS[lvl_idx]]
        classes = list(model.classes_)
        raw_idx = np.argmax(probs, axis=1)
        raw_preds = [classes[i] for i in raw_idx]
        raw_confs = probs[np.arange(n), raw_idx]

        corrected_preds = []
        corrected_confs = []
        mapping = parent_to_child.get(lvl_idx, {})  # parent_to_child indexing matches previous code
        # For each sample, check if raw_pred is a valid child of current_parent
        for i in range(n):
            parent_label = current_parent[i]
            valid_children = mapping.get(parent_label, [])
            pred_label = raw_preds[i]
            conf = float(raw_confs[i])
            if valid_children and pred_label not in valid_children:
                # choose valid child with highest prob (if any)
                # build dict child->prob using classes order
                valid_probs = {}
                for child in valid_children:
                    if child in class_to_index[LEVELS[lvl_idx]]:
                        idx_c = class_to_index[LEVELS[lvl_idx]][child]
                        valid_probs[child] = float(probs[i, idx_c])
                if valid_probs:
                    best = max(valid_probs, key=valid_probs.get)
                    pred_label = best
                    conf = valid_probs[best]
                else:
                    # fallback to first valid child
                    pred_label = valid_children[0]
                    if pred_label in class_to_index[LEVELS[lvl_idx]]:
                        conf = float(probs[i, class_to_index[LEVELS[lvl_idx]][pred_label]])
                    else:
                        conf = conf
            corrected_preds.append(pred_label)
            corrected_confs.append(conf)
        preds_td[lvl_idx] = corrected_preds
        confs_td[lvl_idx] = corrected_confs
        current_parent = corrected_preds

    return preds_td, confs_td


# ----------------------------
# 3) Bottom-up (use probs_all, go from bottom level up)
# ----------------------------
def predict_bottomup_from_probs(probs_all, models, child_to_parent):
    n_levels = len(probs_all)
    n = probs_all[0].shape[0]
    preds_bu = [None] * n_levels
    confs_bu = [None] * n_levels

    # bottom level
    lvl_idx = n_levels - 1
    probs = probs_all[lvl_idx]
    model = models[LEVELS[lvl_idx]]
    classes = list(model.classes_)
    idx = np.argmax(probs, axis=1)
    preds = [classes[i] for i in idx]
    confs = probs[np.arange(n), idx].tolist()
    preds_bu[lvl_idx] = preds
    confs_bu[lvl_idx] = confs

    child_preds = preds  # child prediction used for parent correction

    # go upwards
    for lvl_idx in range(n_levels - 2, -1, -1):  # e.g., 3 -> 0
        probs = probs_all[lvl_idx]
        model = models[LEVELS[lvl_idx]]
        classes = list(model.classes_)
        idx = np.argmax(probs, axis=1)
        raw_preds = [classes[i] for i in idx]
        raw_confs = probs[np.arange(n), idx]

        corrected_preds = []
        corrected_confs = []
        # child_to_parent mapping: keys are child_level_idx (e.g. 1 -> mapping for level 2's children -> parent at level1)
        for i in range(n):
            current_pred = raw_preds[i]
            conf = float(raw_confs[i])
            child_label = child_preds[i]
            valid_parent = child_to_parent.get(lvl_idx + 2, {}).get(child_label)
            if valid_parent is not None and current_pred != valid_parent:
                # override with valid_parent
                current_pred = valid_parent
                # update conf if parent exists in classes ordering
                if valid_parent in class_to_index[LEVELS[lvl_idx]]:
                    idx_parent = class_to_index[LEVELS[lvl_idx]][valid_parent]
                    conf = float(probs[i, idx_parent])
            corrected_preds.append(current_pred)
            corrected_confs.append(conf)

        preds_bu[lvl_idx] = corrected_preds
        confs_bu[lvl_idx] = corrected_confs
        child_preds = corrected_preds  # move up

    return preds_bu, confs_bu


# ----------------------------
# Hybrid: choose per-sample between TD vs BU by mean confidence
# ----------------------------
def hybrid_from_td_bu(preds_td, confs_td, preds_bu, confs_bu):
    df = pd.DataFrame({"text": texts})
    n_levels = len(preds_td)
    for i in range(n_levels):
        lvl = LEVELS[i]
        df[f"lvl{lvl}_td"] = preds_td[i]
        df[f"lvl{lvl}_conf_td"] = confs_td[i]
        df[f"lvl{lvl}_bu"] = preds_bu[i]
        df[f"lvl{lvl}_conf_bu"] = confs_bu[i]

    conf_td_cols = [f"lvl{l}_conf_td" for l in LEVELS]
    conf_bu_cols = [f"lvl{l}_conf_bu" for l in LEVELS]
    df["mean_conf_td"] = df[conf_td_cols].mean(axis=1)
    df["mean_conf_bu"] = df[conf_bu_cols].mean(axis=1)
    df["use_td"] = df["mean_conf_td"] > df["mean_conf_bu"]

    preds_hybrid = []
    for i, lvl in enumerate(LEVELS):
        preds_hybrid.append(np.where(df["use_td"], df[f"lvl{lvl}_td"], df[f"lvl{lvl}_bu"]).tolist())
    return preds_hybrid, df


# ----------------------------
# Evaluation & save 
# ----------------------------
def evaluate_and_save(true_labels, pred_labels, texts, out_dir, strategy_name, runtime_sec):
    os.makedirs(out_dir, exist_ok=True)
    num_levels = len(true_labels)
    metrics = {}

    for lvl in range(num_levels):
        y_true = np.array(true_labels[lvl])
        y_pred = np.array(pred_labels[lvl])
        present = np.unique(y_true)
        acc = accuracy_score(y_true, y_pred)
        p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0, labels=present)
        p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0, labels=present)

        metrics[f"Level_{lvl+1}"] = {
            "Accuracy": acc,
            "Precision_weighted": p_w, "Recall_weighted": r_w, "F1_weighted": f_w,
            "Precision_micro": p_mi, "Recall_micro": r_mi, "F1_micro": f_mi,
            "Precision_macro": p_ma, "Recall_macro": r_ma, "F1_macro": f_ma,
            "Num_Test_Classes": len(present)
        }

        # confusion matrix & class report
        cm = confusion_matrix(y_true, y_pred, labels=present)
        pd.DataFrame(cm, index=[f"T_{c}" for c in present], columns=[f"P_{c}" for c in present]).to_csv(os.path.join(out_dir, f"confusion_{strategy_name}_L{lvl+1}.csv"))
        pd.DataFrame(classification_report(y_true, y_pred, labels=present, output_dict=True, zero_division=0)).transpose().to_csv(os.path.join(out_dir, f"class_report_{strategy_name}_L{lvl+1}.csv"))

    path_acc = sum(all(true_labels[l][i] == pred_labels[l][i] for l in range(num_levels)) for i in range(len(texts))) / len(texts)
    metrics["Overall"] = {
        "Path_Accuracy": path_acc,
        "Runtime_sec": runtime_sec,
        "Runtime_per_sample": runtime_sec / len(texts),
        "Device": DEVICE,
        "Used_Memory_MB": psutil.virtual_memory().used / (1024**2)
    }

    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.to_csv(os.path.join(out_dir, f"metrics_{strategy_name}.csv"))
    print(f"[{strategy_name}] saved to {out_dir} | PathAcc={path_acc:.4f}")
    return df


# ----------------------------
# run all strategies
# ----------------------------
start_ts = time.time()

print("=== Independent predictions (batched) ===")
preds_ind, probs_ind = predict_independent(texts, models, vectorizers, batch_size=BATCH_SIZE)
runtime_ind = time.time() - start_ts
evaluate_and_save(true_labels, preds_ind, texts, os.path.join(SAVE_DIR, "independent"), "independent", runtime_ind)

print("=== Top-down (from cached probs) ===")
preds_td, confs_td = predict_topdown_from_probs(probs_ind, models, parent_to_child)
runtime_td = time.time() - start_ts
evaluate_and_save(true_labels, preds_td, texts, os.path.join(SAVE_DIR, "topdown"), "topdown", runtime_td)

print("=== Bottom-up (from cached probs) ===")
preds_bu, confs_bu = predict_bottomup_from_probs(probs_ind, models, child_to_parent)
runtime_bu = time.time() - start_ts
evaluate_and_save(true_labels, preds_bu, texts, os.path.join(SAVE_DIR, "bottomup"), "bottomup", runtime_bu)

print("=== Hybrid ===")
preds_hyb, df_hyb = hybrid_from_td_bu(preds_td, confs_td, preds_bu, confs_bu)
runtime_hyb = time.time() - start_ts
evaluate_and_save(true_labels, preds_hyb, texts, os.path.join(SAVE_DIR, "hybrid"), "hybrid", runtime_hyb)

# Save per-sample predictions (final)
def save_predictions_csv(all_preds, out_csv):
    # all_preds: list(level)[sample]
    d = {"text": texts}
    for i, lvl in enumerate(LEVELS):
        d[f"true_level_{lvl}"] = true_labels[i]
        d[f"pred_level_{lvl}"] = all_preds[i]
    pd.DataFrame(d).to_csv(out_csv, index=False, encoding="utf-8")
    print("Saved predictions to:", out_csv)

save_predictions_csv(preds_ind, os.path.join(SAVE_DIR, "predictions_independent.csv"))
save_predictions_csv(preds_td, os.path.join(SAVE_DIR, "predictions_topdown.csv"))
save_predictions_csv(preds_bu, os.path.join(SAVE_DIR, "predictions_bottomup.csv"))
save_predictions_csv(preds_hyb, os.path.join(SAVE_DIR, "predictions_hybrid.csv"))

print("\nAll strategies completed. Results in:", SAVE_DIR)



