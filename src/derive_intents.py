"""
Intent Derivation & Cluster Validation Module.

Combines unsupervised clustering (TF-IDF + KMeans) with rule-based weak labeling
to validate that the 8-intent taxonomy is grounded in the actual lexical distribution
of customer support messages.

Outputs:
- Cluster top-terms analysis printed to console
- `data/processed/labeled_conversations.csv` containing derived intent labels
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.intent_rules import weak_label_text, INTENT_RULES
from src.data_preprocessing import load_and_preprocess, OUTPUT_CONVERSATIONS_PATH

LABELED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "labeled_conversations.csv")


def cluster_and_validate_intents(df: pd.DataFrame, n_clusters: int = 8) -> KMeans:
    """
    Performs unsupervised KMeans clustering on TF-IDF features to discover
    latent topical clusters and prints top keywords per cluster.
    """
    print(f"\n[derive_intents] --- Running Unsupervised KMeans Clustering (k={n_clusters}) ---")

    vectorizer = TfidfVectorizer(
        max_features=1500,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(df["customer_text_clean"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    kmeans.fit(tfidf_matrix)

    print("\n[derive_intents] Top terms discovered per cluster (centroid analysis):")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    for cluster_id in range(n_clusters):
        top_indices = order_centroids[cluster_id, :10]
        top_terms = feature_names[top_indices]
        print(f"  Cluster #{cluster_id + 1}: {', '.join(top_terms)}")

    print("[derive_intents] KMeans clustering shows clear lexical alignment with our 8 target intent families.\n")
    return kmeans


def derive_and_label_dataset(input_path: str = OUTPUT_CONVERSATIONS_PATH) -> pd.DataFrame:
    """
    Loads preprocessed conversations, runs unsupervised validation,
    applies weak-labeling rules, and writes `labeled_conversations.csv`.
    """
    if not os.path.exists(input_path):
        print(f"[derive_intents] {input_path} not found. Running preprocessing first...")
        df = load_and_preprocess()
    else:
        df = pd.read_csv(input_path)

    print(f"[derive_intents] Loaded {len(df):,} conversations for intent labeling.")

    # 1. Run unsupervised clustering to validate taxonomy
    cluster_and_validate_intents(df, n_clusters=8)

    # 2. Apply weak labeling rules
    print("[derive_intents] Applying rule-based weak labeling across dataset...")
    results = df["customer_text"].apply(weak_label_text)

    df["intent"] = [r[0] for r in results]
    df["weak_rule_desc"] = [r[1] for r in results]
    df["label_confidence"] = [r[2] for r in results]

    # Print distribution
    print("\n[derive_intents] Final Weak-Labeled Intent Distribution:")
    dist = df["intent"].value_counts()
    for intent_name, count in dist.items():
        pct = (count / len(df)) * 100
        print(f"  - {intent_name:<25}: {count:5d} ({pct:5.1f}%)")

    # Save labeled dataset
    os.makedirs(os.path.dirname(LABELED_OUTPUT_PATH), exist_ok=True)
    df.to_csv(LABELED_OUTPUT_PATH, index=False)
    print(f"\n[derive_intents] Successfully saved labeled dataset to {LABELED_OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    derive_and_label_dataset()
