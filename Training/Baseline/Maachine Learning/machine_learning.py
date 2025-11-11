# %%
import time
import psutil
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from joblib import dump

# Optional GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_available = True
except:
    gpu_available = False
    print("pynvml not found or GPU unavailable. GPU memory tracking skipped.")

BASE_DIR = "your_project_dir" # input project directory
DATA_DIR = os.path.join(BASE_DIR, "your_data_dir")  # input data training directory
MODEL_BASE_PATH = os.path.join(BASE_DIR, "Models/machine_learning")

# ========== 1. Load Dataset ==========
# access a CSV file in the dataset
# adjust the column 'label' based on level
# e.g: for level 2: 'label_level_2' --> 'label' and previous column 'label' --> label_level_1
level = 1
train_df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_data_train_encoded.csv"), sep=',')
train_df.columns = ["Text", "Desc", "label", "label_level_2", "label_level_3", "label_level_4", "label_level_5"]

val_df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_data_val_encoded.csv"), sep=',')
val_df.columns = ["Text", "Desc", "label", "label_level_2", "label_level_3", "label_level_4", "label_level_5"]

test_df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_data_test_encoded.csv"), sep=',')
test_df.columns = ["Text", "Desc", "label", "label_level_2", "label_level_3", "label_level_4", "label_level_5"]

# ========== 2. Vectorizer ==========
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_df["Desc"])
y_train = train_df["label"]

X_val = vectorizer.transform(val_df["Desc"])
y_val = val_df["label"]

X_test = vectorizer.transform(test_df["Desc"])
y_test = test_df["label"]

# ========== 3. Models to Compare ==========
models = {
    "LogisticRegression": LogisticRegression(max_iter=5000),
    "SVM": SVC(kernel='linear', probability=True),
    "NaiveBayes": MultinomialNB()
}

results = []

def get_gpu_memory_usage_MB():
    """total GPU memory usage (MB) from all devices"""
    if not gpu_available:
        return 0.0
    total = 0
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total += mem_info.used / (1024 ** 2)  # bytes -> MB
    return total

# ========== 4. Training & Evaluation ==========
for name, clf in models.items():
    print(f"\n===== Training {name} =====")

    # Monitor memory before training
    cpu_mem_before = psutil.virtual_memory().used / (1024 ** 2)
    gpu_mem_before = get_gpu_memory_usage_MB()

    # --- Training phase ---
    start_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Monitor memory after training
    cpu_mem_after = psutil.virtual_memory().used / (1024 ** 2)
    gpu_mem_after = get_gpu_memory_usage_MB()

    cpu_peak_MB = abs(cpu_mem_after - cpu_mem_before)
    gpu_peak_MB = abs(gpu_mem_after - gpu_mem_before)

    # --- Validation ---
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"{name} Validation Accuracy: {val_acc:.4f}")

    # --- Testing phase ---
    start_test = time.time()
    test_preds = clf.predict(X_test)
    test_time = time.time() - start_test
    avg_time_per_record = test_time / len(test_df)

    # --- Metrics ---
    metrics = {
        "Model": name,
        "Train_time_sec": train_time,
        "Test_time_total_sec": test_time,
        "Avg_time_per_record_sec": avg_time_per_record,
        "Accuracy": accuracy_score(y_test, test_preds),
        "Precision_weighted": precision_score(y_test, test_preds, average="weighted"),
        "Recall_weighted": recall_score(y_test, test_preds, average="weighted"),
        "F1_weighted": f1_score(y_test, test_preds, average="weighted"),
        "Precision_micro": precision_score(y_test, test_preds, average="micro"),
        "Recall_micro": recall_score(y_test, test_preds, average="micro"),
        "F1_micro": f1_score(y_test, test_preds, average="micro"),
        "Precision_macro": precision_score(y_test, test_preds, average="macro"),
        "Recall_macro": recall_score(y_test, test_preds, average="macro"),
        "F1_macro": f1_score(y_test, test_preds, average="macro"),
        "Total_GPU_peak_MB": gpu_peak_MB,
        "Total_CPU_peak_MB": cpu_peak_MB,
    }

    results.append(metrics)
    
    # Save model
    dump(clf, os.path.join(MODEL_BASE_PATH,f"{name}_model_lvl_{level}.joblib"))
    print(f"{name} saved as {name}_model.joblib")

    # Save predictions
    test_df[f"{name}_predicted"] = test_preds
    test_df.to_csv(os.path.join(MODEL_BASE_PATH,f"{name}_test_predictions_lvl_{level}.csv"), index=False)
    print(f"{name} predictions saved to {name}_test_predictions.csv")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))

# ========== 5. Save All Metrics ==========
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODEL_BASE_PATH,"ml_models_comparison_metrics_lvl_{level}.csv"), index=False)
print("\n All metrics saved to ml_models_comparison_metrics.csv")

# Save vectorizer
dump(vectorizer, os.path.join(MODEL_BASE_PATH, "tfidf_vectorizer.joblib"))
print("Vectorizer saved as tfidf_vectorizer.joblib")

if gpu_available:
    pynvml.nvmlShutdown()



