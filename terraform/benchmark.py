import json
import time
import urllib.request
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA = Path("creditcard.csv")
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


if not DATA.exists():
    print(f"Downloading dataset to {DATA} ...", flush=True)
    urllib.request.urlretrieve(DATA_URL, DATA)

started = time.perf_counter()
df = pd.read_csv(DATA)
load_time = time.perf_counter() - started
X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
train_started = time.perf_counter()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
training_time = time.perf_counter() - train_started

probability = model.predict_proba(X_test)[:, 1]
prediction = (probability >= 0.5).astype(int)

single_started = time.perf_counter()
for _ in range(100):
    model.predict_proba(X_test.iloc[[0]])
single_latency_ms = (time.perf_counter() - single_started) / 100 * 1000

batch = X_test.iloc[:1000]
batch_started = time.perf_counter()
model.predict_proba(batch)
batch_seconds = time.perf_counter() - batch_started

results = {
    "dataset_rows": int(len(df)),
    "load_time_seconds": load_time,
    "training_time_seconds": training_time,
    "best_iteration": int(model.best_iteration_),
    "auc_roc": roc_auc_score(y_test, probability),
    "accuracy": accuracy_score(y_test, prediction),
    "f1_score": f1_score(y_test, prediction),
    "precision": precision_score(y_test, prediction),
    "recall": recall_score(y_test, prediction),
    "inference_latency_single_ms": single_latency_ms,
    "inference_time_1000_rows_seconds": batch_seconds,
    "inference_throughput_rows_per_second": 1000 / batch_seconds,
}

Path("benchmark_result.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
print("\n=== LightGBM Credit Card Fraud Benchmark ===")
for key, value in results.items():
    print(f"{key}: {value}")
print("\nSaved benchmark_result.json")
