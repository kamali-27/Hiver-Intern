"""
Model Training Module for Intent Classification.

Trains, evaluates, and persists three classifiers:
1. Baseline 1 — Majority Class Classifier (models/baseline_majority.pkl)
2. Baseline 2 — Standard Unigram TF-IDF + Logistic Regression (models/tfidf_logreg.pkl)
3. Final System Model — Tuned (1,2)-gram Sublinear TF-IDF + Balanced Logistic Regression
   (models/final_intent_classifier.pkl)

Uses stratified Train (70%) / Validation (15%) / Test (15%) splits.
Persists all trained models to the `models/` directory.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LABELED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "labeled_conversations.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class MajorityBaselineModel:
    """Trivial baseline that always predicts the most frequent class."""
    def __init__(self):
        self.majority_class = None
        self.classes_ = None

    def fit(self, X, y):
        counts = pd.Series(y).value_counts()
        self.majority_class = counts.index[0]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array([self.majority_class] * len(X))

    def predict_proba(self, X):
        n_samples = len(X)
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))
        maj_idx = list(self.classes_).index(self.majority_class)
        probas[:, maj_idx] = 1.0
        return probas


def evaluate_model(name: str, model: Any, X_test, y_test) -> Dict[str, float]:
    """Calculates accuracy, precision, recall, and macro/weighted F1."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    metrics = {
        "model": name,
        "accuracy": round(acc, 4),
        "precision_macro": round(p_macro, 4),
        "recall_macro": round(r_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(p_wt, 4),
        "recall_weighted": round(r_wt, 4),
        "f1_weighted": round(f1_wt, 4),
    }
    return metrics


def train_and_evaluate_all():
    """Main training workflow."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(LABELED_DATA_PATH):
        print(f"[train] {LABELED_DATA_PATH} not found. Running intent derivation first...")
        from src.derive_intents import derive_and_label_dataset
        df = derive_and_label_dataset()
    else:
        df = pd.read_csv(LABELED_DATA_PATH)

    print(f"[train] Loaded {len(df):,} labeled rows for training.")

    X = np.array(df["customer_text"].astype(str).tolist(), dtype=object)
    y = np.array(df["intent"].astype(str).tolist(), dtype=object)

    # Stratified Train (70%) / Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Split Temp into Val (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[train] Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # -------------------------------------------------------------
    # 1. Baseline 1: Majority Class
    # -------------------------------------------------------------
    print("\n[train] Training Baseline 1 (Majority Class)...")
    majority_model = MajorityBaselineModel()
    majority_model.fit(X_train, y_train)
    m1_metrics = evaluate_model("Baseline 1 (Majority Class)", majority_model, X_test, y_test)
    joblib.dump(majority_model, os.path.join(MODELS_DIR, "baseline_majority.pkl"))

    # -------------------------------------------------------------
    # 2. Baseline 2: Standard TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("[train] Training Baseline 2 (Standard Unigram TF-IDF + LogReg)...")
    tfidf_logreg = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=1000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=500, random_state=42))
    ])
    tfidf_logreg.fit(X_train, y_train)
    m2_metrics = evaluate_model("Baseline 2 (TF-IDF + LogReg)", tfidf_logreg, X_test, y_test)
    joblib.dump(tfidf_logreg, os.path.join(MODELS_DIR, "tfidf_logreg.pkl"))

    # -------------------------------------------------------------
    # 3. Final System Model: (1,2)-gram Sublinear TF-IDF + Balanced LogReg
    # -------------------------------------------------------------
    print("[train] Training Final System Model (Bigram Sublinear TF-IDF + Balanced Tuned LogReg)...")
    final_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=4000,
            token_pattern=r"(?u)\b\w+\b"
        )),
        ("clf", LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ))
    ])
    final_model.fit(X_train, y_train)
    final_metrics = evaluate_model("Final System (Optimized N-Gram TF-IDF + LogReg)", final_model, X_test, y_test)
    joblib.dump(final_model, os.path.join(MODELS_DIR, "final_intent_classifier.pkl"))

    # Save class label list
    classes = list(final_model.classes_)
    with open(os.path.join(MODELS_DIR, "intent_classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    # -------------------------------------------------------------
    # Summary Comparison Table
    # -------------------------------------------------------------
    results_df = pd.DataFrame([m1_metrics, m2_metrics, final_metrics])
    print("\n" + "="*85)
    print("HELD-OUT TEST SET INTENT CLASSIFICATION BENCHMARK")
    print("="*85)
    print(results_df.to_string(index=False))
    print("="*85)

    # Detailed report for final model
    y_pred_final = final_model.predict(X_test)
    print("\nDetailed Classification Report for Final System Model on Test Set:")
    print(classification_report(y_test, y_pred_final, zero_division=0))

    return results_df


if __name__ == "__main__":
    train_and_evaluate_all()
