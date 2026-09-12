"""
Quantitative Retrieval Benchmark: Sparse (TF-IDF) vs Hybrid (Dense + Sparse RRF).

Evaluates case retrieval grounding across the 194-sample Golden Evaluation Benchmark:
- Precision@1 / Intent Hit Rate @ 1
- Intent Recall @ 3 (Hit Rate @ 3)
- Mean Reciprocal Rank (MRR@3)
- Average Top Precedent Similarity Score
"""

import os
import sys
import time
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval import get_or_build_retriever

GOLDEN_SET_PATH = os.path.join(PROJECT_ROOT, "eval", "golden_eval_set.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "eval", "results", "retrieval_comparison.csv")


def evaluate_retrieval_modes():
    print("=" * 75)
    print("RUNNING RETRIEVAL BENCHMARK: SPARSE TF-IDF vs HYBRID DENSE+SPARSE")
    print("=" * 75)

    df_gold = pd.read_csv(GOLDEN_SET_PATH)
    print(f"[benchmark] Loaded {len(df_gold)} golden evaluation examples.")

    retriever = get_or_build_retriever(load_dense=True)

    metrics = {
        "Sparse (TF-IDF)": {"hit_1": 0, "hit_3": 0, "rr_sum": 0.0, "sim_sum": 0.0, "time_ms": 0.0},
        "Hybrid (Dense + Sparse)": {"hit_1": 0, "hit_3": 0, "rr_sum": 0.0, "sim_sum": 0.0, "time_ms": 0.0}
    }

    n_samples = len(df_gold)

    for mode in ["Sparse (TF-IDF)", "Hybrid (Dense + Sparse)"]:
        internal_mode = "hybrid" if "Hybrid" in mode else "sparse"
        t0 = time.time()
        for _, row in df_gold.iterrows():
            gold_intent = str(row["gold_intent"])
            query = str(row["message"])

            cases = retriever.retrieve(query, k=3, mode=internal_mode)
            if not cases:
                continue

            top_sim = cases[0]["similarity_score"]
            metrics[mode]["sim_sum"] += top_sim

            # Intent hit ranking
            hit_rank = 0
            for r, case in enumerate(cases, 1):
                if case.get("intent") == gold_intent:
                    hit_rank = r
                    break

            if hit_rank == 1:
                metrics[mode]["hit_1"] += 1
            if hit_rank in [1, 2, 3]:
                metrics[mode]["hit_3"] += 1
                metrics[mode]["rr_sum"] += 1.0 / hit_rank

        total_time = (time.time() - t0) * 1000
        metrics[mode]["time_ms"] = total_time / n_samples

    # Summary
    results = []
    for mode, m in metrics.items():
        results.append({
            "Retrieval Mode": mode,
            "Intent Hit@1 (Precision@1)": f"{(m['hit_1'] / n_samples) * 100:.1f}%",
            "Intent Hit@3 (Recall@3)": f"{(m['hit_3'] / n_samples) * 100:.1f}%",
            "Mean Reciprocal Rank (MRR@3)": f"{(m['rr_sum'] / n_samples):.3f}",
            "Avg Top Precedent Similarity": f"{(m['sim_sum'] / n_samples):.3f}",
            "Avg Latency / Query": f"{m['time_ms']:.2f} ms"
        })

    res_df = pd.DataFrame(results)
    print("\n--- BENCHMARK RESULTS ---")
    print(res_df.to_string(index=False))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    res_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved retrieval comparison to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_retrieval_modes()
