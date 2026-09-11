# Why Headline Accuracy and F1 Numbers Are Misleading in Customer Support AI

In academic benchmarks and introductory machine learning presentations, teams frequently celebrate metrics like:
> **"Our final intent classification model achieved 70.1% Accuracy and 70.4% Macro F1!"**

While this represents a substantial gain over the Majority Class baseline (12.9% accuracy) and standard TF-IDF (63.9% accuracy), **relying on this single headline metric to judge the production readiness of a customer support AI system is dangerous and fundamentally misleading**.

Below are the 6 critical reasons why headline numbers fail to reflect real-world support performance and business risk.

---

## 1. Weak-Label Noise & Evaluation Circularity

The underlying training set (`data/processed/labeled_conversations.csv`) was generated using **rule-based weak labeling and TF-IDF KMeans clustering** because the original Twitter customer support dataset contains no human-annotated intent tags.

- **The Circularity Trap**: The ML classifier is trained to mimic keyword-matching rules and clustering centroids. If the rules contained systemic blind spots (e.g., misclassifying replacement requests as refund requests), the model simply learns and reproduces those errors.
- **Inflated Validation Scores**: High validation accuracy on weakly labeled data often just proves that the model successfully memorized the regex patterns used to generate the labels, not that it understands customer semantics.
- **The Golden Set Mitigation**: While our 194-sample Golden Set (`eval/golden_eval_set.csv`) was independently curated to test actual boundaries, even it reflects human heuristics. True ground truth requires blind double-annotation with inter-annotator agreement (Cohen's Kappa > 0.85).

---

## 2. Asymmetric Business Costs: False Auto-Handles vs. False Escalations

Standard accuracy and Macro F1 assign equal penalty to all classification errors. In enterprise customer support, **the cost matrix is wildly asymmetric**:

$$\text{Cost}(\text{False Auto-Handle}) \gg \text{Cost}(\text{False Escalation})$$

| Error Type | What Happens in Practice | Business Consequence | Real Cost |
|---|---|---|---|
| **False Auto-Handle** (Under-Escalation) | A furious customer threatening a lawsuit or reporting fraud receives a generic automated "check your tracking link" reply. | Escalated regulatory risk, public Twitter PR crisis, churn of high-LTV customer. | **Catastrophic ($$$$)** |
| **False Escalation** (Over-Escalation) | A routine tracking query is sent to a human support agent's ticket queue with all metadata extracted. | 90 seconds of agent triage time; customer receives accurate human response. | **Negligible ($)** |

A model with **85% accuracy** that occasionally makes reckless autonomous decisions on volatile queries is commercially unusable. Conversely, a system with **65% accuracy** paired with conservative, explainable safety gates that escalates whenever uncertain is highly valuable in production.

---

## 3. Brand-Specific Narrowness & Domain Overfitting

The dataset and vectorizer are tuned exclusively to `AmazonHelp`:
- **Amazon Specifics**: The vocabulary is dominated by Prime shipping terms (`1-day delivery`, `transit hub`, `locker drop-off`, `carrier marked delivered`, `amzn.to/help`).
- **Transfer Collapse**: Deploying this exact model to another customer support domain (e.g., `AppleSupport` dealing with iOS iCloud recovery or `DeltaAir` dealing with cancelled flight rebooking) would result in near-total accuracy collapse (< 25%).
- **Headline Distortion**: The headline 70.1% score is a measure of in-domain memorization on a single retail giant's tweet distribution, not generalized conversational understanding.

---

## 4. Class Imbalance and the Illusion of Macro F1

In real e-commerce operations, ticket volumes are heavily skewed:
- Logistics queries (`order_delivery_delay`) and financial disputes (`refund_request`) represent **over 65%** of all customer contacts.
- Complex technical bugs (`technical_bug`) or account recovery issues (`account_access`) make up less than **10%**.

Macro F1 weights every class equally, meaning poor performance on rare edge cases can artificially depress the score even if 90% of actual customer volume is handled seamlessly. Conversely, Weighted F1 (70.2%) can mask total failure on a critical minority class like `complaint_escalation`, which represents the highest legal and brand risk.

---

## 5. Limitations of Rubric-Based Heuristic Evaluation

Our zero-API-key evaluation pipeline uses a rubric-based heuristic judge (`eval/llm_judge.py`) measuring:
- Keyword presence
- Message length bounds
- Empathy phrase inclusion
- Grounding token citations

While this enables fast, reproducible local benchmarking without API dependencies, **heuristic judges suffer from gaming**:
1. A generated reply that says: *"Hello! We apologize for your order delay. Please DM us at amzn.to/help with your 17-digit order number so we can investigate."* gets a **5.0/5.0** on relevance and brand consistency—even if the customer actually asked how to close their account!
2. Our human-vs-judge analysis (`eval/results/judge_agreement.md`) confirmed a low Cohen's Kappa (`0.138`), showing that automated heuristics struggle to capture real semantic nuances and factual coherence that human reviewers immediately catch.

---

## 6. Single-Turn Twitter Snippets vs. Real Multi-Turn Dialogues

Public Twitter customer support interactions are fundamentally fragmented:
- **Missing State**: Customers often send fragmented tweets across multiple replies: *"Wait, also it was sent to the wrong address"* or *"Forget my previous tweet, I found it."*
- **Privacy Censorship**: Tweets constantly redirect customers to private Direct Messages (`"Please send us a DM"`). As a result, the training dataset captures only the **initial greeting and deflection**, rarely the actual final resolution.
- **Headline Disconnect**: Evaluating single-turn reply quality gives no indication of whether the agent can successfully resolve a problem across an interactive 5-turn customer dialogue.

---

## Executive Conclusion

> **Key Takeaway for SDEs & Stakeholders:**  
> Never ship or evaluate a customer support agent based on classification accuracy alone. Production readiness is measured by:
> 1. **Zero High-Risk Escapes**: 0% under-escalation on legal, fraud, and severe anger.
> 2. **Precision on High-Confidence Auto-Handles**: > 95% satisfaction on the subset of tickets the agent *chooses* to answer autonomously.
> 3. **Deflection without Rework**: Customers who receive an autonomous reply should not immediately reopen the ticket or demand a supervisor.
