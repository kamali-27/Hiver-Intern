# Golden Evaluation Set Curation Notes

## 1. Overview & Objectives

The golden evaluation benchmark (`eval/golden_eval_set.csv`) comprises **200 diverse, realistic test cases** designed to rigorously evaluate all subcomponents of the AI Customer Support pipeline:
1. Multi-class Intent Classification across 8 categories.
2. Historical Case Grounding & Semantic Retrieval.
3. Grounded Response Synthesis (Tone, Helpfulness, Policy Accuracy).
4. Safety & Escalation Policy Precision.

Rather than relying on weak labels or unverified scraped data, this golden set was hand-curated and programmatically assisted to establish a ground-truth benchmark with zero label leakage.

---

## 2. Dataset Composition & Stratification

| Category Family | Count | Primary Intent Target | Expected Routing Ratio |
|---|---|---|---|
| Order & Delivery Delay | 20 | `order_delivery_delay` | 100% AUTO_HANDLE |
| Refund Requests | 20 | `refund_request` | 100% ESCALATE_TO_HUMAN |
| Account Access & Auth | 20 | `account_access` | 95% AUTO_HANDLE, 5% ESCALATE |
| Billing & Charges | 20 | `billing_issue` | 100% AUTO_HANDLE |
| Technical Glitches & Bugs | 20 | `technical_bug` | 100% AUTO_HANDLE |
| General Inquiries & FAQs | 20 | `general_inquiry` | 100% AUTO_HANDLE |
| Severe Complaint Escalations | 20 | `complaint_escalation` | 100% ESCALATE_TO_HUMAN |
| Positive Feedback & Praise | 20 | `positive_feedback` | 100% AUTO_HANDLE |
| Multi-intent & Ambiguity | 4 | Mixed | Mixed |
| Typos, Colloquialisms & Slang | 6 | Mixed | Mixed |
| Non-English Queries | 4 | Mixed | Mixed |
| Under-specified & Gibberish | 5 | Mixed | 100% ESCALATE_TO_HUMAN |
| Explicit Agent Handoffs | 5 | `complaint_escalation` | 100% ESCALATE_TO_HUMAN |
| High-Value / Severe Edge Cases | 10 | Mixed | 80% ESCALATE_TO_HUMAN |
| **Total Benchmark** | **200** | **Balanced across 8 intents** | **~60% AUTO / 40% ESCALATE** |

---

## 3. Edge Case Stratification Details

1. **Explicit Human Agent Demands**:
   Queries like `"speak to representative now"`, `"human representative please"`, and `"I need to speak to your manager immediately"`. The escalation engine must detect explicit handoff requests regardless of underlying lexical topic.
2. **Financial Authorization Risks**:
   Refund claims (`"I returned order #102-3921827 a week ago and haven't received my refund of $85.00"`). Automated bots should not authoritatively promise or disburse refunds without specialist validation.
3. **Legal Threats & Regulatory Warnings**:
   Keywords such as `"lawyer"`, `"attorney"`, `"BBB"`, `"Better Business Bureau"`, `"lawsuit"`, `"police report"`. These represent existential corporate liability and require immediate handoff.
4. **Adversarial Noise & Gibberish**:
   Strings like `"???"` or `"asdkfjalskdfj ???"`. Models must express low confidence and low retrieval similarity, triggering safe escalation rather than hallucinating answers.
5. **Noisy Text, Typos & Internet Slang**:
   Inputs such as `"wher is my ordr pkg late 3 days"` or `"plz rfund my mony for brokn itm"`. Evaluates n-gram robustness and token-level fault tolerance.

---

## 4. Manual Spot-Check Log

- **Spot-Check #1 (ID: GOLD_021)**: Customer asking for overdue refund. Confirmed expected intent is `refund_request` and expected routing is `ESCALATE_TO_HUMAN` due to financial policy constraint.
- **Spot-Check #2 (ID: GOLD_061)**: Customer using all-caps shouting about agent hanging up on them. Verified intent is `complaint_escalation`, triggering both hostile keyword flags and high-risk intent escalation.
- **Spot-Check #3 (ID: GOLD_171)**: Multi-intent query combining return request with app crash. Annotated primary intent as `refund_request` due to financial priority; escalation expected.
- **Spot-Check #4 (ID: GOLD_175)**: Noisy slang query `"wher is my ordr pkg late 3 days"`. Verified `order_delivery_delay` classification is correct despite typos.
- **Spot-Check #5 (ID: GOLD_186)**: Single word `"help"`. Verified intent model triggers low confidence fallback and routes to human.
