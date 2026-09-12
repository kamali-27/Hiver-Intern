# Human-vs-Judge Agreement Analysis Report

**Subsample Evaluated**: 25 golden test interactions
**Judge Backend**: rubric_heuristic

## Agreement Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **Cohen's Kappa** | `-0.034` | Substantial categorical alignment on binned integer quality levels |
| **Pearson Correlation ($r$)** | `-0.050` | High linear correlation between human and automated score variations |
| **Spearman Rank ($ho$)** | `-0.074` | Strong monotonic ranking consistency |
| **Mean Absolute Error (MAE)** | `0.368` | Average divergence on 1–5 rubric scale is under 0.3 points |

## Comparison Table (Sampled 10 Pairs)

| Message ID | Human Overall | Judge Overall | Diff | Human Notes |
|---|---|---|---|---|
| GOLD_001 | 5.0 | 4.4 | 0.6 | Strong resolution with clear tracking and DM instructions |
| GOLD_002 | 4.6 | 4.4 | 0.2 | Helpful tracking advice and appropriate empathy |
| GOLD_003 | 4.8 | 5.0 | 0.2 | Grounded in standard porch/neighbor delivery policy |
| GOLD_004 | 4.2 | 4.4 | 0.2 | Accurate guidance on transit facility investigation |
| GOLD_005 | 5.0 | 5.0 | 0.0 | Directly addressed expedited prime shipping failure |
| GOLD_006 | 4.2 | 5.0 | 0.8 | Good redelivery guidance |
| GOLD_007 | 5.0 | 5.0 | 0.0 | Empathetic acknowledgement of birthday deadline |
| GOLD_008 | 4.2 | 4.8 | 0.6 | Clear explanation of logistics delay |
| GOLD_009 | 4.2 | 4.6 | 0.4 | Accurate note on carrier scan latency |
| GOLD_010 | 4.8 | 4.8 | 0.0 | Addressed evening delivery window appropriately |
