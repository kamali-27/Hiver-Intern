# One-Week Engineering Improvement Roadmap

If allocated an additional week of dedicated engineering time, here is the prioritized, actionable roadmap designed to advance this system from an initial prototype to an enterprise-grade production platform.

---

## Roadmap Overview (Gantt Summary)

| Day | Focus Area | Deliverables | Expected Impact |
|:---:|---|---|---|
| **Day 1** | **Active Learning & True Human Labels** | 500 uncertainty-sampled golden examples annotated by support leads | Replaces weak-label circularity; +8-10% classification accuracy |
| **Day 2** | **Dense Semantic Retrieval (Hybrid Search)** | Bi-encoder embeddings (`all-MiniLM-L6-v2`) + BM25 Reciprocal Rank Fusion | Grounding relevance jumps from 0.35 to 0.75+ cosine similarity |
| **Day 3** | **Cost-Sensitive Escalation Optimization** | Neyman-Pearson threshold tuning with Bayesian optimization | Reduces false over-escalations by 40% while keeping under-escalation at 0% |
| **Day 4** | **Multi-Turn Thread Reconstruction** | Thread builder merging inbound chains (`in_response_to_tweet_id`) | Eliminates single-turn amnesia; enables stateful slot filling |
| **Day 5** | **Multi-Brand Domain Generalization** | Adapter layers for `AppleSupport`, `SpotifyCares`, and `DeltaAir` | Extends platform applicability across retail, tech, and travel |
| **Day 6** | **Real LLM-as-a-Judge Calibration** | Few-shot GPT-4o / Claude 3.5 Sonnet judge with pairwise Elo | Cohen's Kappa improves from 0.14 to > 0.70 |
| **Day 7** | **Streaming, Observability & A/B Canary Plan** | Prometheus metrics, token streaming, and gradual rollout harness | Sub-250ms p95 latency and safe live shadow traffic validation |

---

## Detailed Daily Engineering Plan

### Day 1: Active Learning Loop & Human Annotation
- **Problem**: The current pipeline trains on rule-based weak labels, limiting accuracy to ~70% and perpetuating label noise.
- **Action**:
  - Implement an **uncertainty sampling pipeline** using prediction entropy:
    $$H(p) = -\sum_{i=1}^K p_i \log_2 p_i$$
  - Sample the top 500 customer queries where intent confidence falls in the ambiguous zone ($0.35 \le \text{confidence} \le 0.60$).
  - Deploy a simple Streamlit annotation UI for internal tier-2 support staff to label verified intents and slot entities.
- **Success Metric**: Retrain intent classifier on verified human data; target > 82% macro F1.

### Day 2: Hybrid Retrieval (Dense Semantic + Sparse Lexical)
- **Problem**: TF-IDF cosine similarity fails on synonyms and paraphrase variations (e.g., "where is my stuff" vs "package tracking location").
- **Action**:
  - Integrate a lightweight local bi-encoder (`sentence-transformers/all-MiniLM-L6-v2`, ~80MB, fast CPU inference).
  - Precompute 384-dimensional dense vectors for all 2,500 historical conversations.
  - Implement **Reciprocal Rank Fusion (RRF)** combining dense semantic search with sparse BM25:
    $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + r_m(d)}$$
  - Add strict intent-filtering metadata so retrieved cases are guaranteed to match the inquiry domain.
- **Success Metric**: Mean Reciprocal Rank (MRR@3) increases from 0.42 to > 0.80.

### Day 3: Asymmetric Cost-Sensitive Escalation Optimization
- **Problem**: Current escalation rules use static thresholds (0.55 confidence, 0.25 similarity), causing a 50.5% over-escalation rate on benign queries.
- **Action**:
  - Formulate escalation as a **constrained Neyman-Pearson optimization problem**:
    $$\min \text{Escalation Rate} \quad \text{subject to} \quad P(\text{Under-Escalate} \mid \text{High-Risk}) \le 0.005$$
  - Implement dynamic, class-conditioned thresholds:
    - Routine delivery tracking: threshold = `0.42`
    - Technical bugs / generic FAQ: threshold = `0.50`
    - Financial refunds / account security: threshold = `0.70`
- **Success Metric**: Reduce unnecessary human escalation volume by 40% without increasing customer risk.

### Day 4: Multi-Turn Conversation State Tracking
- **Problem**: Customer service on social media spans 3-6 turns; evaluating single tweets ignores conversation context and prior agent promises.
- **Action**:
  - Write a graph traversal script on `twcs.csv` using `tweet_id` and `in_response_to_tweet_id` to reconstruct complete multi-turn conversation trees.
  - Store dialog history in an in-memory session store (or Redis).
  - Implement state tracking variables: `order_id_collected`, `verification_completed`, `refund_status_checked`.
- **Success Metric**: The agent successfully answers follow-up inquiries like *"yes, that's my order number"* without requesting information twice.

### Day 5: Multi-Brand Cross-Domain Adaptation
- **Problem**: The system is specialized for Amazon retail. Deploying to other enterprise clients requires generalizable architecture.
- **Action**:
  - Ingest 50,000 tweets across three distinct verticals:
    - **Retail**: `AmazonHelp`
    - **Consumer Tech / Hardware**: `AppleSupport`
    - **Subscription / SaaS**: `SpotifyCares`
  - Implement a two-tiered classification hierarchy:
    - Tier 1: Domain-Agnostic Intent (`billing`, `auth`, `outage`, `complaint`)
    - Tier 2: Domain-Specific Sub-intent (`icloud_sync`, `prime_delivery`, `playlist_offline`)
- **Success Metric**: Zero-shot cross-brand transfer accuracy > 60%; fine-tuned transfer > 85%.

### Day 6: Real LLM-as-a-Judge Calibration & Elo Arena
- **Problem**: Rubric heuristic judges can be gamed by length and formatting, showing weak agreement with human reviewers (Kappa = 0.138).
- **Action**:
  - Deploy a calibrated LLM-judge pipeline using structured outputs (JSON schema) with explicit rubrics and 3-shot few-shot examples.
  - Implement **Pairwise Comparison (Glicko-2 / Elo rating)** between generated responses and actual human agent replies.
  - Benchmark automated judge ratings against 100 human-annotated samples until Cohen's Kappa exceeds 0.70.
- **Success Metric**: Automated judge evaluation closely mirrors expert human quality audits.

### Day 7: Streaming UI, Observability, and Shadow Canary Deployment
- **Problem**: Real-time customer experience requires low latency (< 300ms) and telemetry.
- **Action**:
  - Upgrade Streamlit interface to stream token responses via Python generators.
  - Add structured Prometheus/OpenTelemetry logging:
    - `inference_latency_seconds_bucket`
    - `intent_prediction_distribution_total`
    - `escalation_trigger_reasons_counter`
  - Build a **Shadow Traffic Canary harness**: the AI agent scores live incoming customer tickets in the background and compares its recommendations to human agent actions before taking autonomous control.
- **Success Metric**: p95 pipeline latency under 250ms on CPU; complete observability dashboard ready for production operations.
