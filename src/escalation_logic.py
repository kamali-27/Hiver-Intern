"""
Escalation Logic Engine.

Interpretable, rule-based decision system that evaluates whether an incoming
customer support interaction can be safely handled autonomously (`AUTO_HANDLE`)
or requires immediate human operator intervention (`ESCALATE_TO_HUMAN`).

Evaluates 5 orthogonal signals:
1. Low classifier confidence (< 0.55)
2. High-risk business intent (refund_request, complaint_escalation)
3. Low retrieval grounding similarity (< 0.25)
4. Hostile sentiment, legal threats, or fraud allegations
5. Explicit requests for human agent, representative, or supervisor
"""

import re
from typing import Dict, Any, List

# Tunable thresholds documented in docs/decision_log.md
CONFIDENCE_THRESHOLD = 0.55
SIMILARITY_THRESHOLD = 0.25

HIGH_RISK_INTENTS = {
    "complaint_escalation",
    "refund_request"
}

HOSTILE_PATTERNS = [
    r"\b(sue|lawyer|attorney|lawsuit|legal\s+action)\b",
    r"\b(scam|scammers|fraud|stole\s+my\s+money|thieves)\b",
    r"\b(better\s+business\s+bureau|bbb|police|ftc|authorities)\b",
    r"\b(worst\s+(customer\s+)?service|disgraceful|gross\s+negligence)\b",
    r"\b(hung\s+up\s+on\s+me|rude\s+agent|lied\s+to\s+me)\b"
]

EXPLICIT_HUMAN_PATTERNS = [
    r"\b(speak|talk)\s+to\s+(a\s+)?(human|person|agent|rep|representative|manager|supervisor)\b",
    r"\b(want|need|give\s+me)\s+(a\s+)?(real\s+person|human|supervisor|manager)\b",
    r"\b(transfer\s+(me\s+to\s+a\s+)?(human|representative|agent))\b",
    r"\b(agent\s+please|customer\s+service\s+rep)\b"
]


class EscalationEngine:
    """
    Evaluates customer interaction signals to decide routing policy.
    """
    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold

    def evaluate(
        self,
        message: str,
        predicted_intent: str,
        intent_confidence: float,
        top_retrieval_similarity: float
    ) -> Dict[str, Any]:
        """
        Returns structured decision dict:
        - decision: 'AUTO_HANDLE' or 'ESCALATE_TO_HUMAN'
        - reason: Human-readable explanation of the routing decision
        - flags: List of triggered rule flags
        """
        flags: List[str] = []
        reasons: List[str] = []

        msg_lower = (message or "").lower().strip()

        # Signal 1: Explicit Human Agent Request
        for pattern in EXPLICIT_HUMAN_PATTERNS:
            if re.search(pattern, msg_lower):
                flags.append("EXPLICIT_HUMAN_REQUEST")
                reasons.append("Customer explicitly requested a human agent or supervisor.")
                break

        # Signal 2: Hostility, Legal Threats, Fraud Accusations
        for pattern in HOSTILE_PATTERNS:
            match = re.search(pattern, msg_lower)
            if match:
                flags.append("HOSTILE_OR_LEGAL_THREAT")
                reasons.append(f"Detected elevated risk keywords/threats ('{match.group(0)}').")
                break

        # Signal 3: High-Risk Intent Category
        if predicted_intent in HIGH_RISK_INTENTS:
            flags.append(f"HIGH_RISK_INTENT:{predicted_intent.upper()}")
            if predicted_intent == "complaint_escalation":
                reasons.append("Severe customer dissatisfaction requires human empathy and resolution.")
            elif predicted_intent == "refund_request":
                reasons.append("Financial transactions and refund authorizations require specialist review.")

        # Signal 4: Low Intent Classification Confidence
        if intent_confidence < self.confidence_threshold:
            flags.append("LOW_INTENT_CONFIDENCE")
            reasons.append(
                f"Intent confidence ({intent_confidence:.2f}) is below safe threshold ({self.confidence_threshold:.2f})."
            )

        # Signal 5: Low Historical Case Similarity (Grounding Failure Risk)
        if top_retrieval_similarity < self.similarity_threshold:
            flags.append("LOW_HISTORICAL_SIMILARITY")
            reasons.append(
                f"Top historical precedent similarity ({top_retrieval_similarity:.2f}) is below grounding threshold ({self.similarity_threshold:.2f})."
            )

        # Decision synthesis
        if flags:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": " | ".join(reasons),
                "flags": flags
            }
        else:
            return {
                "decision": "AUTO_HANDLE",
                "reason": (
                    f"Routine query ({predicted_intent}) with high intent confidence ({intent_confidence:.2f}) "
                    f"and strong historical case grounding ({top_retrieval_similarity:.2f}). Safe for autonomous response."
                ),
                "flags": []
            }


if __name__ == "__main__":
    engine = EscalationEngine()

    test_cases = [
        ("Where is order 123? It was supposed to be here yesterday.", "order_delivery_delay", 0.91, 0.76),
        ("Give me a human right now, your bot is useless!", "complaint_escalation", 0.88, 0.65),
        ("I returned the headphones last week, please refund my $50.", "refund_request", 0.85, 0.72),
        ("I am suing your company for fraud and filing a BBB complaint.", "complaint_escalation", 0.94, 0.58),
        ("Um hello maybe some thing happened", "general_inquiry", 0.32, 0.12),
        ("Can I use gift cards to buy kindle ebooks?", "general_inquiry", 0.88, 0.64)
    ]

    for msg, intent, conf, sim in test_cases:
        res = engine.evaluate(msg, intent, conf, sim)
        print(f"[{res['decision']}] Reason: {res['reason']}")
        print(f"  Input: '{msg}'\n")
