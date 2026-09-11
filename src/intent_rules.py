"""
Intent Rules & Weak Labeling Engine.

Defines explicit keyword, regex, and linguistic heuristics across the 8-intent taxonomy.
Used to bootstrap weak labels on unannotated customer service tweets.
"""

import re
from typing import Tuple, Dict, List

# Explicit intent taxonomy definitions
INTENT_RULES: Dict[str, Dict[str, List[str]]] = {
    "order_delivery_delay": {
        "keywords": [
            "deliver", "delivery", "shipping", "shipped", "package", "parcel", "transit",
            "tracking", "carrier", "late", "delay", "delayed", "where is my", "not received",
            "haven't received", "hasn't arrived", "lost", "stuck", "front door", "porch",
            "eta", "dispatch", "dispatched", "courier", "attempted delivery", "missing package"
        ],
        "patterns": [
            r"\b(where\s+(is|are)\s+my\s+(order|package|item|shipment))\b",
            r"\b(still\s+(haven't|not|hasn't)\s+(arrived|received|delivered))\b",
            r"\b(tracking\s+(says|number|id|status|link))\b",
            r"\b(delivery\s+(delay|delayed|late|attempted))\b",
            r"\b(package\s+(lost|stuck|missing))\b"
        ]
    },
    "refund_request": {
        "keywords": [
            "refund", "refunds", "refunded", "money back", "return", "returned", "returning",
            "cancel order", "cancel my", "cancellation", "reimburse", "reimbursement",
            "compensation", "send back", "drop off locker", "return label", "credit note"
        ],
        "patterns": [
            r"\b(want|need|give\s+me|demand)\s+my\s+money\s+back\b",
            r"\b(process|issue|where\s+is)\s+(my\s+)?refund\b",
            r"\b(haven't|never|not)\s+(got|received)\s+(my\s+)?refund\b",
            r"\b(returned|sent\s+back)\s+(the\s+)?(item|order|product)\b",
            r"\b(cancel\s+(my\s+)?order)\b"
        ]
    },
    "account_access": {
        "keywords": [
            "login", "log in", "log-in", "signin", "sign in", "sign-in", "password",
            "passcode", "otp", "2fa", "two-factor", "verification code", "locked out",
            "locked account", "compromised", "hacked", "auth", "authenticator", "reset link",
            "suspicious activity", "cannot access", "unauthorized access", "old phone number"
        ],
        "patterns": [
            r"\b(locked\s+out\s+of\s+my\s+account)\b",
            r"\b(can't|cannot|unable\s+to)\s+(log\s*in|sign\s*in|access\s+my\s+account)\b",
            r"\b(reset|forgot)\s+(my\s+)?password\b",
            r"\b(otp|verification\s+code|2fa)\s+(not|never)\s+(arrived|received|sent|working)\b",
            r"\b(account\s+(hacked|compromised|blocked|suspended|disabled))\b"
        ]
    },
    "billing_issue": {
        "keywords": [
            "charged", "charge", "charges", "billing", "bill", "billed", "credit card",
            "debit card", "duplicate charge", "double charged", "overcharged", "unauthorized charge",
            "invoice", "receipt", "subscription", "membership fee", "prime fee", "declined",
            "payment failed", "payment declined", "auto-renew", "auto renew", "bank statement"
        ],
        "patterns": [
            r"\b(charged\s+twice|double\s+charg(ed|ing)|duplicate\s+charge)\b",
            r"\b(unauthorized|unexpected|extra)\s+charge\b",
            r"\b(charged\s+(me\s+)?\$?\d+)\b",
            r"\b(payment\s+(was\s+)?declined|payment\s+failed)\b",
            r"\b(membership|subscription|prime)\s+(fee|charge|renew(ed|al)?)\b"
        ]
    },
    "technical_bug": {
        "keywords": [
            "crash", "crashes", "crashing", "bug", "glitch", "error", "error 500", "error 404",
            "freeze", "frozen", "freezes", "blank screen", "white screen", "not loading",
            "won't load", "app", "website", "checkout button", "cart empty", "unresponsive",
            "stutter", "buffering", "server error", "broken", "update failed", "looping"
        ],
        "patterns": [
            r"\b(app\s+(keeps\s+)?crash(es|ing)?)\b",
            r"\b(website|page|screen)\s+(is\s+)?(frozen|broken|blank|unresponsive)\b",
            r"\b(error\s+(code\s+)?\d{3}|internal\s+server\s+error)\b",
            r"\b(can't|cannot|won't)\s+(click|add\s+to\s+cart|load|checkout)\b",
            r"\b(glitch|bug\s+in\s+the\s+app)\b"
        ]
    },
    "complaint_escalation": {
        "keywords": [
            "worst customer service", "terrible service", "horrible service", "unacceptable",
            "disgraceful", "manager", "supervisor", "representative", "human", "real person",
            "attorney", "lawyer", "legal action", "lawsuit", "sue", "scam", "fraud", "scammers",
            "stole", "steal", "thieves", "better business bureau", "bbb", "consumer protection",
            "hung up", "rude agent", "useless support", "complaint", "escalate"
        ],
        "patterns": [
            r"\b(speak|talk)\s+to\s+a\s+(human|person|agent|representative|manager|supervisor)\b",
            r"\b(worst\s+(customer\s+)?(service|support|experience))\b",
            r"\b(take|taking)\s+legal\s+action\b",
            r"\b(report(ing)?\s+(this\s+to|you\s+to)\s+(the\s+)?(bbb|police|authorities|ftc))\b",
            r"\b(this\s+is\s+(an\s+absolute\s+)?(scam|fraud|unacceptable|theft))\b",
            r"\b(agent\s+(hung\s+up|was\s+rude|lied))\b"
        ]
    },
    "positive_feedback": {
        "keywords": [
            "thank you", "thanks", "appreciate", "kudos", "shoutout", "great service",
            "amazing support", "awesome", "fantastic", "five stars", "5 stars", "best support",
            "helpful", "grateful", "speedy resolution", "solved quickly", "lifesaver",
            "love you guys", "wonderful service", "super fast"
        ],
        "patterns": [
            r"\b(thank(s|\s+you)\s+(so\s+much|very\s+much|a\s+lot)?)\b",
            r"\b(great|amazing|awesome|fantastic|excellent)\s+(customer\s+)?(service|support|job|help)\b",
            r"\b(shoutout\s+to|kudos\s+to)\b",
            r"\b(solved|resolved)\s+(my\s+issue\s+)?(in\s+\d+\s+minutes|quickly|fast)\b",
            r"\b(5|five)\s+stars\b"
        ]
    },
    "general_inquiry": {
        "keywords": [
            "question", "inquiry", "how do i", "how can i", "is it possible", "warranty",
            "international shipping", "gift card", "policy", "return policy", "available",
            "stock", "discount", "student discount", "store hours", "price match", "specifications"
        ],
        "patterns": [
            r"\b(how\s+(do|can)\s+i\s+(find|change|know|use|apply))\b",
            r"\b(do\s+you\s+(ship|offer|have|support|sell))\b",
            r"\b(what\s+is\s+the\s+(policy|warranty|difference|price))\b",
            r"\b(is\s+(it|there)\s+possible\s+to)\b"
        ]
    }
}

# Priority ordering when multiple rules match:
# Escalations & refunds take precedence over general order inquiries
INTENT_PRIORITY = [
    "complaint_escalation",
    "refund_request",
    "account_access",
    "billing_issue",
    "technical_bug",
    "positive_feedback",
    "order_delivery_delay",
    "general_inquiry"
]


def weak_label_text(text: str) -> Tuple[str, str, float]:
    """
    Evaluates customer tweet against rule sets.
    Returns: (predicted_intent, matched_rule_description, rule_confidence_score)
    """
    if not isinstance(text, str) or not text.strip():
        return ("general_inquiry", "empty_fallback", 0.1)

    text_lower = text.lower()
    matches: Dict[str, Dict] = {}

    for intent, rules in INTENT_RULES.items():
        keyword_hits = [kw for kw in rules["keywords"] if kw in text_lower]
        pattern_hits = [p for p in rules["patterns"] if re.search(p, text_lower)]

        score = len(keyword_hits) * 1.0 + len(pattern_hits) * 2.5
        if score > 0:
            matches[intent] = {
                "score": score,
                "keywords": keyword_hits,
                "patterns": pattern_hits
            }

    if not matches:
        return ("general_inquiry", "no_rule_match_default", 0.3)

    # Sort matches by score first, then by priority hierarchy
    sorted_intents = sorted(
        matches.keys(),
        key=lambda it: (matches[it]["score"], -INTENT_PRIORITY.index(it)),
        reverse=True
    )

    best_intent = sorted_intents[0]
    best_info = matches[best_intent]
    evidence = []
    if best_info["patterns"]:
        evidence.append(f"regex:{best_info['patterns'][0][:30]}")
    if best_info["keywords"]:
        evidence.append(f"keywords:{','.join(best_info['keywords'][:3])}")
    rule_desc = "; ".join(evidence) if evidence else "rule_match"

    confidence = min(0.95, 0.50 + 0.10 * best_info["score"])
    return (best_intent, rule_desc, round(confidence, 2))


def match_intent_rules(text: str) -> Tuple[str, float]:
    """
    Evaluates customer text against rules and returns (predicted_intent, rule_confidence_score).
    """
    intent, _, conf = weak_label_text(text)
    return intent, conf


if __name__ == "__main__":
    test_samples = [
        "Where is my package? Order #102-3921932 was supposed to arrive yesterday!",
        "I demand an immediate refund for order 55219, item arrived broken.",
        "Can't log in, OTP verification code is not being sent to my phone.",
        "Why did you charge my card $14.99 twice this morning?",
        "App keeps crashing when I hit checkout button on iOS 18.",
        "Your service is a total scam. I want to speak to a manager right now!",
        "Thank you so much Sarah for fixing my issue in 5 minutes! 5 stars!",
        "Do you ship electronics internationally to France?"
    ]
    for sample in test_samples:
        intent, rule, conf = weak_label_text(sample)
        print(f"[{intent}] (conf={conf}) -> {sample[:60]}... | {rule}")
