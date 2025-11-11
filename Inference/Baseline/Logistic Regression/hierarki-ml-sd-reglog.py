# %%
import os, time, json, ast, psutil, joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = "your_project_dir" # input project directory
DATA_DIR = os.path.join(BASE_DIR, "your_data_dir")  # input data training directory
MODEL_BASE_PATH = os.path.join(BASE_DIR, "Models/machine_learning")
RESULT_DIR = os.path.join(BASE_DIR, "Results")

JSON_HIER_PATH = os.path.join(DATA_DIR, "parent_to_child.json")
TEST_CSV = os.path.join(DATA_DIR, "synthetic_data_test_encoded.csv") 

# Template paths for per-level models/vectorizers (adjust names/dirs if needed)
MODEL_PATH_TEMPLATE = os.path.join(MODEL_BASE_PATH, "LogisticRegression_model_level_{lvl}.joblib")
VECTORIZER_PATH_TEMPLATE = os.path.join(MODEL_BASE_PATH,"tfidf_vectorizer.joblib")

SAVE_DIR = os.path.join(RESULT_DIR,"hier_out_logreg")

os.makedirs(SAVE_DIR, exist_ok=True)

LEVELS = ["1", "2", "3", "4", "5"]
DEVICE = "CPU"  

# ----------------------------
# LOAD HIERARCHY
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
for lvl_idx, mapping in parent_to_child.items():
    child_level = lvl_idx + 1
    inv_map = {}
    for parent, children in mapping.items():
        for c in children:
            inv_map[c] = parent
    child_to_parent[child_level] = inv_map


# ----------------------------
# LOAD DATA
# ----------------------------
data = pd.read_csv(TEST_CSV)
texts = data["cleaned_deskripsi"].tolist()
true_labels = [data[f"label_level_{lvl}"].tolist() for lvl in LEVELS]
n_samples = len(texts)
print(f"Loaded {n_samples} samples.")

# ----------------------------
# LOAD MODELS & VECTORIZERS
# ----------------------------
models = [joblib.load(MODEL_PATH_TEMPLATE.format(lvl=lvl)) for lvl in LEVELS]
vectorizers = [joblib.load(VECTORIZER_PATH_TEMPLATE.format(lvl=lvl)) for lvl in LEVELS]


# ============================================================
# INFERENCE FUNCTIONS (optimized for batch)
# ============================================================

def predict_independent(texts, models, vectorizers):
    preds, logits = [], []
    for model, vec in zip(models, vectorizers):
        X = vec.transform(texts)
        probs = model.predict_proba(X)
        preds_lvl = np.argmax(probs, axis=1)
        logits.append(np.log(np.clip(probs, 1e-10, 1.0)))
        preds.append([model.classes_[p] for p in preds_lvl])
    return preds, logits


def fallback_topdown(pred, conf, current_parent, parent_to_child, probs, level):
    valid_children = parent_to_child.get(level, {}).get(current_parent, [])
    if not valid_children:
        return pred, conf
    if pred in valid_children:
        return pred, conf
    valid_probs = {c: probs[c] for c in valid_children if c < len(probs)}
    if valid_probs:
        best = max(valid_probs, key=valid_probs.get)
        return best, valid_probs[best]
    return valid_children[0], probs[valid_children[0]]


def fallback_bottomup(pred_parent, conf, current_child, child_to_parent, probs, level):
    valid_parent = child_to_parent.get(level + 1, {}).get(current_child)
    if not valid_parent:
        return pred_parent, conf
    return valid_parent, probs[pred_parent]


def predict_topdown(texts, models, vectorizers, parent_to_child):
    n_levels = len(models)
    preds_all, confs_all = [[] for _ in range(n_levels)], [[] for _ in range(n_levels)]

    # Level 1 independent
    X = vectorizers[0].transform(texts)
    probs = models[0].predict_proba(X)
    preds_lvl = np.argmax(probs, axis=1)
    confs_lvl = probs[np.arange(len(probs)), preds_lvl]
    preds_all[0] = [models[0].classes_[p] for p in preds_lvl]
    confs_all[0] = confs_lvl.tolist()

    current_parent = preds_all[0]
    # Next levels
    for lvl in range(1, n_levels):
        X = vectorizers[lvl].transform(texts)
        probs = models[lvl].predict_proba(X)
        preds_lvl = np.argmax(probs, axis=1)
        preds_list, conf_list = [], []
        for i in range(len(texts)):
            pred_label = models[lvl].classes_[preds_lvl[i]]
            conf = probs[i, preds_lvl[i]]
            pred_label, conf = fallback_topdown(
                pred_label, conf, current_parent[i], parent_to_child, 
                {c: probs[i, j] for j, c in enumerate(models[lvl].classes_)}, lvl
            )
            preds_list.append(pred_label)
            conf_list.append(conf)
        preds_all[lvl] = preds_list
        confs_all[lvl] = conf_list
        current_parent = preds_list
    return preds_all, confs_all


def predict_bottomup(texts, models, vectorizers, child_to_parent):
    n_levels = len(models)
    preds_all = [[] for _ in range(n_levels)]
    confs_all = [[] for _ in range(n_levels)]

    # --- Lowest level (Level 5)
    X = vectorizers[-1].transform(texts)
    probs = models[-1].predict_proba(X)
    preds_idx = np.argmax(probs, axis=1)
    preds_all[-1] = [models[-1].classes_[i] for i in preds_idx]
    confs_all[-1] = probs[np.arange(len(probs)), preds_idx].tolist()

    child_pred_from_lower_level = preds_all[-1]

    # --- Loop from Level 4 to Level 1 (reverse)
    for i in reversed(range(n_levels - 1)):  # i = 3 → 0
        lvl_name = str(i + 1)
        model = models[i]
        X = vectorizers[i].transform(texts)
        probs = model.predict_proba(X)
        preds_idx = np.argmax(probs, axis=1)
        preds_lvl = [model.classes_[p] for p in preds_idx]
        confs_lvl = probs[np.arange(len(probs)), preds_idx]

        corrected_preds, corrected_confs = [], []
        for j in range(len(texts)):
            current_pred = preds_lvl[j]
            conf = confs_lvl[j]
            child_label = child_pred_from_lower_level[j]

            valid_parent = child_to_parent.get(i + 2, {}).get(child_label)
            if valid_parent is not None and current_pred != valid_parent:
                current_pred = valid_parent
                if valid_parent in model.classes_:
                    idx_parent = list(model.classes_).index(valid_parent)
                    conf = probs[j, idx_parent]
                else:
                    conf = conf  

            corrected_preds.append(current_pred)
            corrected_confs.append(conf)

        preds_all[i] = corrected_preds
        confs_all[i] = corrected_confs
        child_pred_from_lower_level = corrected_preds

    return preds_all, confs_all


def hybrid_from_td_bu(preds_td, confs_td, preds_bu, confs_bu):
    df = pd.DataFrame()
    n_levels = len(preds_td)
    for i in range(n_levels):
        lvl = LEVELS[i]
        df[f"lvl{lvl}_td"] = preds_td[i]
        df[f"lvl{lvl}_bu"] = preds_bu[i]
        df[f"lvl{lvl}_conf_td"] = confs_td[i]
        df[f"lvl{lvl}_conf_bu"] = confs_bu[i]
    df["mean_conf_td"] = df[[f"lvl{l}_conf_td" for l in LEVELS]].mean(axis=1)
    df["mean_conf_bu"] = df[[f"lvl{l}_conf_bu" for l in LEVELS]].mean(axis=1)
    df["use_td"] = df["mean_conf_td"] > df["mean_conf_bu"]

    preds_hybrid = []
    for i, lvl in enumerate(LEVELS):
        preds_hybrid.append(np.where(df["use_td"], df[f"lvl{lvl}_td"], df[f"lvl{lvl}_bu"]).tolist())
    return preds_hybrid, df


# ============================================================
# METRIC & SAVE FUNCTIONS
# ============================================================

def evaluate_and_save(true_labels, pred_labels, texts, out_dir, strategy_name, runtime_sec):
    os.makedirs(out_dir, exist_ok=True)
    num_levels = len(true_labels)
    metrics = {}

    for lvl in range(num_levels):
        y_true, y_pred = np.array(true_labels[lvl]), np.array(pred_labels[lvl])
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
    print(f"[{strategy_name}] PathAcc={path_acc:.4f}")
    return df


# ============================================================
# RUN STRATEGIES
# ============================================================
start_all = time.time()

preds_ind, logits_ind = predict_independent(texts, models, vectorizers)
runtime_ind = time.time() - start_all
evaluate_and_save(true_labels, preds_ind, texts, os.path.join(SAVE_DIR, "independent"), "independent", runtime_ind)

preds_td, confs_td = predict_topdown(texts, models, vectorizers, parent_to_child)
evaluate_and_save(true_labels, preds_td, texts, os.path.join(SAVE_DIR, "topdown"), "topdown", runtime_ind)

preds_bu, confs_bu = predict_bottomup(texts, models, vectorizers, child_to_parent)
evaluate_and_save(true_labels, preds_bu, texts, os.path.join(SAVE_DIR, "bottomup"), "bottomup", runtime_ind)

preds_hyb, df_hyb = hybrid_from_td_bu(preds_td, confs_td, preds_bu, confs_bu)
evaluate_and_save(true_labels, preds_hyb, texts, os.path.join(SAVE_DIR, "hybrid"), "hybrid", runtime_ind)

print("\n All strategies done. Results saved under:", SAVE_DIR)



