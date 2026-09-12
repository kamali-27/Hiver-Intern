"""
Multi-Turn Customer Support Simulation Engine.

Provides:
1. Multi-turn conversation state tracking (DialogTurn, SimulatorState).
2. Customer sentiment and frustration trajectory tracking (-1.0 to +1.0).
3. Slot/entity extraction (order IDs, tracking numbers, dispute amounts).
4. Pre-built stress-test customer personas and scenario scripts:
   - Late Delivery Escalation
   - Unauthorized Double Billing
   - Damaged Item & Refund Dispute
   - Account 2FA Lockout
   - Hostile Legal & BBB Threat
   - Routine FAQ Inquiries
5. Stateful pipeline processing with cumulative multi-turn risk escalation.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.pipeline import CustomerSupportPipeline, get_pipeline


@dataclass
class DialogTurn:
    turn_index: int
    customer_message: str
    agent_reply: str
    predicted_intent: str
    intent_confidence: float
    escalation_decision: str
    escalation_reason: str
    escalation_flags: List[str]
    similarity_score: float
    sentiment_score: float
    frustration_level: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SimulatorState:
    conversation_id: str
    scenario_name: str
    customer_name: str
    turns: List[DialogTurn] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    cumulative_frustration: float = 0.0
    current_status: str = "ACTIVE"  # ACTIVE, AUTO_RESOLVED, ESCALATED_TO_HUMAN, CLOSED
    escalation_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "scenario_name": self.scenario_name,
            "customer_name": self.customer_name,
            "total_turns": len(self.turns),
            "extracted_entities": self.extracted_entities,
            "cumulative_frustration": round(self.cumulative_frustration, 2),
            "current_status": self.current_status,
            "escalation_triggered": self.escalation_triggered
        }


# Lexicon for fast rule-based sentiment & frustration estimation
NEGATIVE_TRIGGERS = {
    "horrible": -0.8, "worst": -0.9, "terrible": -0.8, "awful": -0.8,
    "angry": -0.7, "mad": -0.6, "furious": -0.9, "frustrated": -0.7,
    "ruined": -0.8, "unacceptable": -0.8, "scam": -0.9, "fraud": -0.9,
    "lawyer": -0.9, "sue": -0.9, "bbb": -0.8, "police": -0.9,
    "stole": -0.9, "thieves": -0.9, "lying": -0.8, "ridiculous": -0.7,
    "delayed": -0.4, "late": -0.4, "waiting": -0.3, "stuck": -0.4,
    "broken": -0.6, "shattered": -0.7, "damaged": -0.6, "useless": -0.7,
    "dispute": -0.7, "overcharged": -0.7, "unauthorized": -0.8, "refund": -0.4
}

POSITIVE_TRIGGERS = {
    "thanks": 0.5, "thank you": 0.6, "great": 0.6, "good": 0.5,
    "appreciate": 0.6, "helpful": 0.6, "perfect": 0.8, "awesome": 0.7,
    "resolved": 0.7, "sorted": 0.6, "pleased": 0.6, "glad": 0.5
}


def estimate_sentiment(text: str) -> float:
    """Estimates sentiment polarity from -1.0 (extremely hostile) to +1.0 (very positive)."""
    text_lower = (text or "").lower()
    score = 0.0
    matched = 0

    for word, weight in NEGATIVE_TRIGGERS.items():
        if word in text_lower:
            score += weight
            matched += 1

    for word, weight in POSITIVE_TRIGGERS.items():
        if word in text_lower:
            score += weight
            matched += 1

    if matched == 0:
        return 0.0

    return max(-1.0, min(1.0, round(score / max(1, matched), 2)))


def determine_frustration_level(sentiment: float) -> str:
    """Classifies sentiment into discrete operational frustration levels."""
    if sentiment <= -0.6:
        return "CRITICAL / HOSTILE"
    elif sentiment <= -0.2:
        return "FRUSTRATED"
    elif sentiment < 0.2:
        return "NEUTRAL / INQUIRING"
    else:
        return "SATISFIED / POLITE"


def extract_entities_from_text(text: str) -> Dict[str, str]:
    """Extracts order numbers, tracking IDs, currency amounts, and account emails."""
    entities = {}

    # Order ID regex: e.g. #112-9281726 or #112-9281726-1234567
    order_match = re.search(r"#?(\d{3}-\d{7}(?:-\d{7})?)", text)
    if order_match:
        entities["order_id"] = order_match.group(1)

    # Tracking number regex: e.g. TBA123456789 or 1Z9999999999999999
    tracking_match = re.search(r"\b(TBA\d{10,12}|1Z[0-9A-Z]{16})\b", text, re.I)
    if tracking_match:
        entities["tracking_number"] = tracking_match.group(1)

    # Currency amount regex: e.g. $120.00, $72.49
    currency_match = re.search(r"(\$\s?\d+(?:\.\d{2})?)", text)
    if currency_match:
        entities["dispute_amount"] = currency_match.group(1)

    # Email regex
    email_match = re.search(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b", text)
    if email_match:
        entities["customer_email"] = email_match.group(1)

    return entities


# Predefined Realistic Stress-Test Scenarios
STRESS_SCENARIOS = {
    "late_delivery": {
        "title": "📦 Scenario 1: Late Delivery Escalation",
        "description": "Customer inquires politely about order tracking, but escalates to extreme frustration as the delay threatens a flight.",
        "customer_name": "Sarah Jenkins (@sarah_j88)",
        "turns": [
            "Hi, can you tell me where order #112-9281726 is? Tracking hasn't updated since yesterday.",
            "It was supposed to arrive 2 days ago! Tracking still says 'in transit'. What is the delay?",
            "I have a flight tomorrow morning and this package contains my passport holder and travel gear! This is completely unacceptable!",
            "I need to speak to a supervisor or human representative right now! Your service has ruined my trip!"
        ]
    },
    "billing_dispute": {
        "title": "💳 Scenario 2: Unauthorized Double Billing",
        "description": "Customer notices duplicate Prime charges and demands an immediate credit card reversal.",
        "customer_name": "David Marcus (@d_marcus)",
        "turns": [
            "I noticed an unexpected duplicate charge of $14.99 on my credit card statement for Prime Video yesterday.",
            "I checked my bank account again and there are TWO separate charges on Sept 10. Why was I charged twice for the same month?",
            "Do not credit this as a gift card! I demand a full refund back to my Visa card immediately or I will dispute the charge with my bank."
        ]
    },
    "damaged_return": {
        "title": "📦 Scenario 3: Shattered Item & Refund Delay",
        "description": "Customer received a broken ceramic dining set, dropped it at an Amazon Locker, and is waiting for refund.",
        "customer_name": "Elena Rostova (@elena_r)",
        "turns": [
            "The ceramic dinner set for order #402-8827162 arrived completely shattered in pieces.",
            "I dropped the return package off at the Amazon Locker on 5th street 4 days ago. Return tracking says received.",
            "Where is my $120.00 refund? The app still hasn't updated the refund status."
        ]
    },
    "account_lockout": {
        "title": "🔐 Scenario 4: 2FA Account Lockout",
        "description": "Customer changed phone carriers and is unable to receive two-step verification codes to access their account.",
        "customer_name": "Marcus Vance (@mvance_tech)",
        "turns": [
            "I am locked out of my Amazon account because the 2-factor authentication code is going to my old phone number.",
            "I already tried clicking 'need help signing in' but it asks for an OTP sent to that same disconnected phone!",
            "Can a customer service specialist please verify my identity via my email marcus.vance@example.com and reset 2FA?"
        ]
    },
    "legal_threat": {
        "title": "🚨 Scenario 5: Hostile Legal & BBB Threat",
        "description": "Customer claims expensive jewelry was stolen during transit and immediately threatens legal action.",
        "customer_name": "Robert Sterling (@r_sterling_esq)",
        "turns": [
            "Your driver marked order #902-1182741 as handed to resident, but nobody was home and my porch camera shows the driver stealing the box!",
            "This package had a $1,200 watch inside! You thieves stole my money. I am filing a police report, contacting my attorney, and reporting Amazon to the Better Business Bureau today!"
        ]
    },
    "routine_faq": {
        "title": "❓ Scenario 6: Routine FAQ & Return Policy",
        "description": "Customer asks standard pre-purchase and return window questions that can be autonomously resolved.",
        "customer_name": "Chloe Adams (@chloe_a)",
        "turns": [
            "Hi there! What is the holiday return window for electronics purchased in October?",
            "Great, thanks! And do I need to keep the original packaging if I need to return it to a Kohl's or Whole Foods drop-off?"
        ]
    }
}


class SimulatorEngine:
    """
    Executes and tracks multi-turn customer dialog simulations.
    """
    def __init__(self, pipeline: Optional[CustomerSupportPipeline] = None):
        self.pipeline = pipeline or get_pipeline()

    def create_session(self, scenario_key: str) -> SimulatorState:
        """Initializes a new simulation session for a selected scenario."""
        scenario = STRESS_SCENARIOS.get(scenario_key, STRESS_SCENARIOS["late_delivery"])
        import uuid
        conv_id = f"SIM_{uuid.uuid4().hex[:8].upper()}"

        return SimulatorState(
            conversation_id=conv_id,
            scenario_name=scenario["title"],
            customer_name=scenario["customer_name"]
        )

    def process_turn(self, state: SimulatorState, customer_message: str, retrieval_mode: str = "hybrid") -> DialogTurn:
        """
        Processes a single customer message in the ongoing multi-turn dialog,
        updating state, cumulative frustration, and routing status.
        """
        t0 = time.time()
        res = self.pipeline.process_message(customer_message, retrieval_mode=retrieval_mode)
        latency = (time.time() - t0) * 1000

        # Sentiment analysis
        sentiment = estimate_sentiment(customer_message)
        frustration = determine_frustration_level(sentiment)

        # Cumulative frustration accumulation
        if sentiment < 0:
            state.cumulative_frustration = min(1.0, state.cumulative_frustration + abs(sentiment) * 0.4)
        else:
            state.cumulative_frustration = max(0.0, state.cumulative_frustration - sentiment * 0.3)

        # Extract entities
        new_entities = extract_entities_from_text(customer_message)
        state.extracted_entities.update(new_entities)

        # Multi-Turn Guardrail: If customer has reached extreme cumulative frustration (>0.75) over multiple turns,
        # override auto-handle and deterministically escalate to human specialist
        decision = res["escalation_decision"]
        reason = res["escalation_reason"]
        flags = list(res["escalation_flags"])

        if state.cumulative_frustration >= 0.75 and decision == "AUTO_HANDLE":
            decision = "ESCALATE_TO_HUMAN"
            reason = f"Cumulative multi-turn customer frustration ({state.cumulative_frustration:.2f}) exceeded threshold. Transferring to human specialist."
            flags.append("MULTI_TURN_FRUSTRATION_ESCALATION")

        # Update session status
        if decision == "ESCALATE_TO_HUMAN":
            state.current_status = "ESCALATED_TO_HUMAN"
            state.escalation_triggered = True
        elif state.current_status != "ESCALATED_TO_HUMAN":
            state.current_status = "AUTO_RESOLVED"

        turn = DialogTurn(
            turn_index=len(state.turns) + 1,
            customer_message=customer_message,
            agent_reply=res["generated_reply"],
            predicted_intent=res["predicted_intent"],
            intent_confidence=res["intent_confidence"],
            escalation_decision=decision,
            escalation_reason=reason,
            escalation_flags=flags,
            similarity_score=res["top_retrieval_similarity"],
            sentiment_score=sentiment,
            frustration_level=frustration,
            latency_ms=round(latency, 2)
        )

        state.turns.append(turn)
        return turn
