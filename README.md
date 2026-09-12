# Autonomous AI Customer Support Agent (Hiver SDE Style)

A production-grade, reproducible, locally runnable AI Customer Support Agent for e-commerce and retail helpdesks, built using Twitter Customer Support data (`AmazonHelp`). 

The system features end-to-end data preprocessing, unsupervised and rule-based intent taxonomy discovery, multi-baseline intent classification, historical case retrieval via TF-IDF vector similarity, template-grounded response generation (100% local, zero paid API keys required), and an explainable 5-tier escalation engine that deterministically separates safe autonomous resolutions from tickets requiring human specialist review.

---

## 1. System Architecture

```
+-----------------------------------------------------------------------------------+
|                        Incoming Customer Support Message                          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                          1. Text Normalization & Cleaning                         |
|                     (URLs, @handles, whitespace, case normalization)              |
+-----------------------------------------------------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
+------------------------------------------+  +-------------------------------------+
|         2. Intent Classification         |  |    3. Historical Case Retrieval     |
|   - Baseline 1: Majority Class (12.9%)   |  |   - Dual-Index: Dense + Sparse      |
|   - Baseline 2: Unigram LogReg (63.9%)   |  |   - all-MiniLM-L6-v2 + TF-IDF RRF   |
|   - Final Model: N-Gram LogReg (70.1%)   |  |   - Cosine & RRF Precedents (k=3)   |
|   Outputs: predicted intent + confidence |  |   Outputs: top cases + similarity   |
+------------------------------------------+  +-------------------------------------+
                      \                                       /
                       \                                     /
                        v                                   v
+-----------------------------------------------------------------------------------+
|                           4. Explainable Escalation Engine                        |
|   Evaluates 5 Deterministic Safety Gates:                                         |
|   1. Intent Confidence Floor (< 0.55)                                             |
|   2. High-Risk Category (`refund_request`, `complaint_escalation`, etc.)          |
|   3. Historical Grounding Precedent (< 0.25 similarity)                           |
|   4. Human Trigger Words ("supervisor", "manager", "lawyer", "fraud", "bbb")      |
|   5. Extreme Frustration / Profanity Lexicon                                      |
|   Outputs: AUTO_HANDLE or ESCALATE_TO_HUMAN + Explicit Audit Reason               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       5. Grounded Support Response Generator                      |
|   - Default Mode: Template + historical brand reply phrase extraction with slot   |
|     filling (order IDs, tracking links, support URLs) and grounding citations.    |
|   - Optional Mode: `USE_LLM=true` hook for local HuggingFace or cloud LLMs.       |
|   Outputs: Outbound reply text + `grounding_sources` [conversation IDs]           |
+-----------------------------------------------------------------------------------+
```

---

## 2. Quickstart & Installation

The core system runs **100% locally** on standard Python 3.10+ without GPU acceleration or external database services.

### Step 1: Clone and Set Up Virtual Environment
```bash
# Clone the repository
cd "customer support"

# Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Dataset & Automatic Fallback Handling

The system targets the **"Customer Support on Twitter"** dataset from Kaggle (`thoughtvector/customer-support-on-twitter`, file `twcs.csv`).

- **Real Dataset Ingestion**: Place your downloaded `twcs.csv` into `data/raw/twcs.csv`. Alternatively, run `python data/download_data.py` to fetch it via `kagglehub` or Kaggle CLI.
- **Automatic Fallback Guarantee**: If `data/raw/twcs.csv` is not present, `data/make_sample_data.py` will automatically generate a realistic 2,500-record synthetic sample mirroring the exact Kaggle schema. **No manual downloads or API keys are required to run the pipeline.**

---

## 4. Exact Reproduction Commands (Step-by-Step)

Execute each step sequentially from the project root:

```bash
# 1. Clean raw customer-brand tweet pairs and extract AmazonHelp conversations
python src/data_preprocessing.py

# 2. Inspect topic clusters via KMeans and generate weak-labeled dataset
python src/derive_intents.py

# 3. Train Baseline 1, Baseline 2, and Final Intent Classifier with model artifacts
python src/train_intent_classifier.py

# 4. Generate the 194-sample Golden Evaluation Set across intents and edge cases
python eval/build_golden_set.py

# 5. Run end-to-end evaluation suite (models, pipeline, heuristic judge, agreement)
python eval/run_evaluation.py

# 6. Benchmark Sparse vs. Hybrid Dense-Sparse Case Retrieval
python eval/eval_retrieval.py

# 7. Run full automated unit & integration test suite (31 tests across 6 test modules)
python -m unittest discover tests

# 8. Launch the interactive Streamlit demonstration web interface
streamlit run app/streamlit_app.py
```

---

## 5. Headline Evaluation Benchmark Results

Evaluated on the independently curated **194-sample Golden Evaluation Set** (`eval/golden_eval_set.csv`):

### 5.1 Intent Classification Benchmark

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1: Majority Class** | 12.89% | 1.61% | 12.50% | 2.85% | 2.94% | Trivial Prior |
| **Baseline 2: TF-IDF + LogReg** | 63.92% | 72.13% | 64.26% | 63.89% | 63.76% | Standard Baseline |
| **Final System: Tuned N-Gram + LogReg** | **70.10%** | **74.04%** | **70.50%** | **70.44%** | **70.21%** | **Measurable Gain (+6.2% F1)** |

### 5.2 Case Retrieval Grounding Benchmark (Sparse vs. Hybrid RRF)

Benchmarking case precedent grounding across 194 golden test queries:

| Retrieval Architecture | Intent Hit@1 (P@1) | Intent Hit@3 (Recall@3) | MRR@3 | Top Similarity | Latency / Query |
|---|:---:|:---:|:---:|:---:|:---:|
| **Sparse (TF-IDF Lexical)** | 54.6% | 60.3% | 0.570 | 0.300 | 7.12 ms |
| **Hybrid (Dense + Sparse RRF)** | **66.0%** | **75.3%** | **0.704** | **0.413** | 244.76 ms (CPU) |
| **Performance Gain** | **+11.4% P@1** | **+15.0% Recall** | **+0.134 MRR** | **+37.7% Sim** | Resolves Paraphrase |

### 5.3 Response Quality Judge Scores (Scale 1–5, Heuristic Rubric)
- **Relevance**: `4.28 / 5.0`
- **Correctness**: `5.00 / 5.0`
- **Grounding**: `4.97 / 5.0`
- **Helpfulness**: `4.41 / 5.0`
- **Brand Consistency**: `4.20 / 5.0`
- **Overall Quality Score**: **`4.57 / 5.0`**

### 5.4 Human-vs-Judge Alignment
- **Subsample Evaluated**: 25 golden conversations
- **Mean Absolute Error (MAE)**: `0.368` points
- **Cohen's Kappa**: Detailed analysis in `eval/results/judge_agreement.md`

### 5.5 Escalation & Safety Routing Policy Benchmark

Evaluates cost-sensitive risk routing across all 194 golden test queries:

| Metric | Measured Value | Operational Meaning |
|---|:---:|---|
| **Routing Policy Accuracy** | **70.62%** | Overall alignment with expected human specialist triage |
| **Under-Escalation Rate** | **0.00%** | **0 / 63 high-risk tickets escaped autonomous handling** (100% recall on legal, fraud, financial disputes) |
| **Over-Escalation Rate** | **43.51%** | 57 / 131 benign routine inquiries sent to human review (down from 50.5% via class-tuned floors & precedent boost) |
| **Auto-Handle Precision** | **100.00%** | Every single autonomously resolved ticket was verified 100% safe |
| **Escalation Recall** | **100.00%** | Zero critical leakage to autonomous response |

---

## 6. Execution Runtime

- **Target Constraint**: Under 15 minutes (900 seconds) on a laptop CPU.
- **Actual Measured Runtime**: **`57.78 seconds`** (0.96 minutes) for the complete end-to-end evaluation suite including hybrid bi-encoder inference across all 194 samples.
- **Runtime Proof**: Verified via wall-clock timing logged at the conclusion of `eval/run_evaluation.py`.

---

## 7. Interactive Streamlit Helpdesk Suite (5 Modules)

Launch with `streamlit run app/streamlit_app.py` to explore:

1. **🛡️ Autonomous Agent Console**:
   - Single-query interactive testing with 7 one-click scenario presets (`Delivery Delay`, `Supervisor Escalation`, `Refund Request`, `Billing Issue`, `Account Lockout`, etc.).
   - Live retrieval engine toggle: **Hybrid (Dense MiniLM + TF-IDF RRF)** vs. **Sparse (TF-IDF)**.
   - Prominent color-coded safety badges (`AUTO_HANDLE` vs. `ESCALATE_TO_HUMAN`) with audit reason and triggered safety flags.
   - Live intent probability decomposition across all 8 classes.
   - Top-3 historical precedent evidence cards with similarity breakdowns.

2. **⚡ Live Support Dialog Simulator**:
   - Multi-turn conversational simulation with pre-built customer persona stress tests.
   - Stateful entity extraction tracking (order IDs, tracking numbers, dispute amounts, customer emails).
   - Turn-by-turn cumulative frustration index tracking with automatic sentiment polarity shifts.
   - Interactive turn advancement (`Advance Next Turn`, `Run All Turns`, `Reset Simulation`).

3. **📊 Brand Operations & Intelligence Dashboard**:
   - Executive KPI cards (Inquiries Analyzed, Auto-Resolution Rate, Safe Precision, Under-Escalation, Monthly Cost Savings).
   - Interactive Altair visualizations: Customer Inquiry Volume by Intent, Routing Escalation Breakdown, and Friction Keyword Frequencies.
   - Support cost efficiency calculator ($6.00/human ticket vs $0.002/AI response).

4. **📚 RAG Knowledge Base & Precedent Search**:
   - Hybrid semantic (all-MiniLM-L6-v2) and lexical (TF-IDF) knowledge retrieval engine over 2,400+ historical Amazon support resolutions.
   - Official SOP policy cheatsheets (Refund SLAs, Lost in Transit, 2FA Recovery, Return Policy).
   - Real-time search query filtering by intent domain, similarity scoring, and on-the-fly grounded response drafting.

5. **📥 Enterprise Support Ticket Inbox**:
   - Helpdesk ticket triage queue (Hiver/Zendesk style) with dynamic queue filtering (All, Auto-Resolved, Human Specialist Queue, High Urgency Financial & Legal).
   - Split-view triage interface: Ticket selection list on left, detail drawer with full audit reasoning on right.
   - Human-in-the-loop action controls: Approve AI Reply, Send Custom Response, Escalate to Tier-2 Specialist, Close Ticket, and internal audit notes trail.

---

## 8. In-Depth Documentation Index

Explore the comprehensive documentation suite in [`docs/`](docs/):
- **[`docs/failure_analysis.md`](docs/failure_analysis.md)**: Deep dive into the top 5 failure modes with real examples, root causes, and production mitigations.
- **[`docs/misleading_headline_number.md`](docs/misleading_headline_number.md)**: Critical analysis detailing why headline accuracy/F1 numbers are fragile and misleading in enterprise support operations.
- **[`docs/one_week_improvements.md`](docs/one_week_improvements.md)**: Concrete, prioritized 7-day engineering roadmap (active learning, hybrid retrieval, multi-brand adaptation).
- **[`docs/decision_log.md`](docs/decision_log.md)**: 16 numbered technical decisions with detailed engineering rationales.

---

## 9. Known Limitations

> [!WARNING]
> While the final intent classifier achieves 70.1% accuracy on the golden benchmark, headline accuracy in customer support AI masks critical risks—such as weak-label noise, severe cost asymmetry between false auto-handles and false escalations, single-brand domain narrowness, and single-turn Twitter fragmentation. Please review **[`docs/misleading_headline_number.md`](docs/misleading_headline_number.md)** for a thorough critical breakdown before deploying in production environments.

