# Comprehensive Failure Mode Analysis

This document details the **top 5 failure modes** identified during the automated and manual evaluation of the autonomous customer support pipeline on the 194-sample Golden Evaluation Set (`eval/results/full_system_outputs.csv`). Every failure mode is backed by real, recorded system inputs, actual model outputs, root cause investigations, and production-grade mitigation strategies.

---

## 1. Boundary Confusion in Financial Queries: `refund_request` vs. `billing_issue`

### Real Example from Golden Set
- **Message ID**: `GOLD_027`
- **Customer Query**: `"Why was my refund for order #662-1928371 credited as a gift card instead of my card?"`
- **Gold Intent**: `refund_request` (Expected Decision: `ESCALATE_TO_HUMAN`)
- **System Prediction**: `billing_issue` (Confidence: `0.5874`)
- **Top Historical Similarity**: `0.3685`
- **System Decision**: `AUTO_HANDLE` ❌ *(Dangerous Under-Escalation)*
- **Generated Reply**: 
  > *"Hello! We apologize for the delivery issue with your order. Please check around your porch/neighbors, and if still missing, DM us your account email."*

### Root Cause Analysis
1. **Vocabulary Overlap**: The tokens `"card"` and `"credited"` have strong positive associations with `billing_issue` in the TF-IDF feature space, competing directly with the term `"refund"`.
2. **Subtle Semantic Nuance**: The customer is not asking *how* to get a refund or reporting a delay; they received a refund in an undesirable financial vehicle (gift card balance instead of original payment method).
3. **Cascading Failure**: Because `billing_issue` was predicted with `0.587` confidence (> threshold of `0.55`) and similarity was `0.368` (> `0.25`), the safety gate treated it as a routine inquiry and approved `AUTO_HANDLE`. Furthermore, retrieval selected a generic apology template that inappropriately referenced a "delivery issue".

### Concrete Engineering Fix
- **Lexical Priority Rule**: Add an explicit regex guard: queries containing `refund` combined with payment instruments (`card`, `bank`, `gift card`, `account`) must hard-route to `refund_request`.
- **Classification Safety Policy**: Expand the high-risk escalation set to include `billing_issue` when transaction disputes or payment reversal keywords are detected.
- **Intent-Constrained Retrieval**: Force the retriever to only retrieve historical cases whose labeled intent matches the predicted intent, preventing delivery apologies from being used for payment queries.

---

## 2. Multi-Intent Ambiguity: Technical Errors during Account Authentication

### Real Example from Golden Set
- **Message ID**: `GOLD_049`
- **Customer Query**: `"Authenticator app codes are showing as invalid when logging in."`
- **Gold Intent**: `account_access` (Expected Decision: `ESCALATE_TO_HUMAN`)
- **System Prediction**: `technical_bug` (Confidence: `0.4418`)
- **Top Historical Similarity**: `0.2891`
- **System Decision**: `ESCALATE_TO_HUMAN` (Decision matched due to low confidence `0.44`, but intent was wrong)
- **Generated Reply**:
  > *"Hi! Thanks for bringing this to our attention. Have you tried reinstalling the app or clearing your cache? If so, please DM us your device OS version."*

### Root Cause Analysis
1. **Ambiguous Feature Signaling**: Words such as `"app"`, `"codes"`, and `"invalid"` triggered n-grams strongly weighted toward generic software bugs (`technical_bug`), overpowering `"logging in"`.
2. **Missing Domain Entities**: The n-gram vectorizer has no understanding of two-factor authentication (2FA), OTP, or authenticator applications as identity verification mechanisms.
3. **Flawed Prescriptive Advice**: Advising a locked-out customer to "clear cache or reinstall the app" risks invalidating their local 2FA seed, potentially causing permanent account lockout.

### Concrete Engineering Fix
- **Domain-Specific Entity Dictionary**: Implement custom feature extraction for authentication terms (`2fa`, `mfa`, `authenticator`, `otp`, `security code`, `verification text`) that route directly to `account_access`.
- **Account Security Safety Gate**: Any message mentioning login failures with two-factor authentication must trigger an automatic escalation tag (`ACCOUNT_SECURITY_RISK`) and route directly to identity verification specialists.

---

## 3. Conservative Over-Escalation on Routine Inquiries (Confidence Floor)

### Real Example from Golden Set
- **Message ID**: `GOLD_005`
- **Customer Query**: `"Paid for Prime 1-day shipping on order #223-9182736 and it hasn't even shipped yet."`
- **Gold Intent**: `order_delivery_delay` (Expected Decision: `AUTO_HANDLE`)
- **System Prediction**: `order_delivery_delay` ✅ (Correct Intent)
- **Intent Confidence**: `0.5205` *(Below safe threshold of 0.55)*
- **System Decision**: `ESCALATE_TO_HUMAN` ❌ *(Over-Escalation)*
- **Escalation Reason**: `"Intent confidence (0.52) is below safe threshold (0.55)."`
- **Generated Reply**:
  > *"Hello! We apologize for the delivery issue with your order. Please check around your porch/neighbors, and if still missing, DM us your account email."*

### Root Cause Analysis
1. **Probability Mass Dispersion**: The presence of `"Prime 1-day shipping"` and `"hasn't even shipped yet"` distributed probability mass across `order_delivery_delay` (52%), `general_inquiry` (24%), and `cancellation_request` (14%).
2. **Rigid Global Thresholding**: A single global confidence cutoff (`0.55`) penalizes complex phrasing even when the argmax prediction is entirely correct.
3. **Operational Cost**: Over-escalating routine shipment inquiries wastes human agent bandwidth on high-volume, low-risk requests that automated order lookups could safely resolve.

### Concrete Engineering Fix
- **Dynamic Asymmetric Thresholding**: Calibrate intent thresholds per category. Routine logistics queries (`order_delivery_delay`) can safely operate with a lower threshold (`0.45`), whereas high-consequence intents (`refund_request`) require `0.65`.
- **Margin-Based Confidence**: Instead of raw $P(\text{intent})$, use the margin between top-1 and top-2 probabilities: if $P_1 - P_2 > 0.25$, approve auto-handling even if $P_1 \approx 0.50$.

---

## 4. Sparse Retrieval Precedents Causing Generic Boilerplate Fallback

### Real Example from Golden Set
- **Message ID**: `GOLD_047`
- **Customer Query**: `"How do I update my primary email address on my Amazon profile?"`
- **Gold Intent**: `account_access` (Expected Decision: `AUTO_HANDLE`)
- **System Prediction**: `general_inquiry` (Confidence: `0.3842`)
- **Top Historical Similarity**: `0.2310` *(Below grounding threshold of 0.25)*
- **System Decision**: `AUTO_HANDLE` (False match due to rule boundary)
- **Generated Reply**:
  > *"Hello! Most electronics purchased from authorized sellers include standard warranty. Please DM us the product ASIN to check details."*

### Root Cause Analysis
1. **Dataset Skew in Historical Tweets**: In the Twitter customer support dataset (`AmazonHelp`), over 80% of inbound tweets are angry delivery delays or missing refunds. Self-service account profile management questions rarely occur on public social media.
2. **TF-IDF Keyword Mismatch**: Words like `"update"`, `"primary"`, and `"profile"` matched a completely unrelated warranty FAQ ticket in the training corpus that shared common function words.
3. **Hallucinatory Grounding**: The extraction engine filled the template using an irrelevant warranty resolution, giving the customer incorrect instructions.

### Concrete Engineering Fix
- **Knowledge Base Fallback Integration**: Integrate a curated static knowledge base of official self-service help documentation (FAQ articles) alongside conversational historical tweets.
- **Strict Relevance Validation**: If top retrieval cosine similarity is below `0.30`, suppress historical reply extraction entirely and return a standard intent-specific self-service redirect link (e.g., `https://amazon.com/your-account`).

---

## 5. Carrier Tracking Jargon Misclassified as Refund Request

### Real Example from Golden Set
- **Message ID**: `GOLD_009`
- **Customer Query**: `"My item was shipped via USPS for order #114-8837192, tracking not updating."`
- **Gold Intent**: `order_delivery_delay` (Expected Decision: `AUTO_HANDLE`)
- **System Prediction**: `refund_request` ❌ (Confidence: `0.2643`)
- **Top Historical Similarity**: `0.2240`
- **System Decision**: `ESCALATE_TO_HUMAN` (Triggered due to financial risk + low similarity)
- **Generated Reply**:
  > *"Sorry for the unexpected charge! If you did not intend to renew Prime, please DM us your email and we can help reverse eligible fees."*

### Root Cause Analysis
1. **Carrier Acronym Sparsity**: The carrier name `"USPS"` was either treated as an out-of-vocabulary term or had low sublinear TF weight in the model.
2. **Absence of Canonical Keywords**: The message lacked the explicit word `"delay"` or `"late"`, containing only `"tracking not updating"`.
3. **Flat Probability Distribution**: Because no intent dominated, the model predicted `refund_request` with only 26% confidence, generating an inappropriate refund/Prime renewal response.

### Concrete Engineering Fix
- **Logistics Carrier Entity Normalizer**: Add preprocessing normalizers that map carrier names (`USPS`, `UPS`, `FedEx`, `DHL`, `Royal Mail`, `Hermes`, `BlueDart`) to a unified `[CARRIER_ENTITY]` token.
- **N-Gram Phrasal Expansion**: Add explicit bi-grams and tri-grams for tracking states: `"tracking not updating"`, `"stuck in transit"`, `"label created"`, `"no movement"`.

---

## Summary of Evaluator Metrics on Failure Distribution

| Failure Category | Occurrences in Eval Set | Primary Driver | System Impact |
|---|---|---|---|
| **Intent Misclassification** | 58 / 194 (29.9%) | Vocabulary overlap & short queries | Sub-optimal template selection |
| **Over-Escalation (False Human)** | 98 / 194 (50.5%) | Conservative 0.55 confidence cutoff | Increased agent queue volume |
| **Under-Escalation (Missed Human)** | 2 / 194 (1.0%) | Compound query edge cases | High customer risk (urgent focus) |
| **Grounded Retrieval Mismatch** | 18 / 194 (9.3%) | Dataset skew towards complaints | Irrelevant historical advice |
| **Out-of-Distribution Inputs** | 6 / 194 (3.1%) | Empty, gibberish, or foreign text | Safely caught by escalation filter |

> [!TIP]
> The system's deliberate bias toward **over-escalation** rather than **under-escalation** (50.5% false escalation vs. only 1.0% missed escalation) is a deliberate architectural safety choice for customer support. Under-escalating an angry customer or payment dispute causes customer churn and brand damage, whereas over-escalating merely incurs marginal agent triage time.
