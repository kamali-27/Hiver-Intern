"""
LLM & Heuristic Evaluation Judge.

Scores generated customer service replies on a 1–5 rubric scale across 5 dimensions:
1. Relevance: Does the reply directly address the customer's stated issue?
2. Correctness: Is the guidance procedurally sound and free of misinformation?
3. Grounding: Is the text firmly anchored in historical precedent without hallucination?
4. Helpfulness: Does the response provide clear, actionable next steps?
5. Brand Consistency: Does the tone reflect professional, empathetic brand voice?

Supports two execution backends:
- Default: Deterministic Rubric-Based Heuristic Judge (100% reproducible, zero API cost)
- Optional: Generative LLM-as-Judge (when `USE_LLM=true` and API key is configured)
"""

import os
import re
from typing import Dict, Any, List, Optional

# Intent-specific expected resolution keywords for correctness/relevance checks
INTENT_RESOLUTION_CRITERIA = {
    "order_delivery_delay": ["track", "carrier", "dm", "delay", "order", "delivery", "investigate", "package"],
    "refund_request": ["refund", "return", "business days", "dm", "warehouse", "order", "status"],
    "account_access": ["security", "password", "dm", "verification", "2fa", "recovery", "account"],
    "billing_issue": ["charge", "transaction", "card", "dm", "billing", "amount", "inspect"],
    "technical_bug": ["cache", "app", "browser", "update", "device", "incognito", "reinstall", "dm"],
    "general_inquiry": ["help", "policy", "amzn.to", "information", "details", "dm"],
    "complaint_escalation": ["apologize", "sorry", "supervisor", "senior", "experience", "dm", "review"],
    "positive_feedback": ["thank", "welcome", "compliments", "glad", "day", "appreciate"]
}


class SupportEvaluationJudge:
    """
    Evaluator for Customer Support Response Quality.
    """
    def __init__(self, use_llm: Optional[bool] = None):
        if use_llm is None:
            env_val = os.getenv("USE_LLM", "false").lower()
            self.use_llm = env_val in ("true", "1", "yes")
        else:
            self.use_llm = use_llm

    def _score_heuristic(
        self,
        customer_message: str,
        predicted_intent: str,
        generated_reply: str,
        retrieved_cases: List[Dict[str, Any]],
        grounding_sources: List[str]
    ) -> Dict[str, Any]:
        """
        Rubric-based heuristic scoring engine.
        """
        reply_lower = (generated_reply or "").lower()
        msg_lower = (customer_message or "").lower()

        # -------------------------------------------------------------
        # 1. Relevance Score (1-5)
        # -------------------------------------------------------------
        # Checks overlap with expected intent vocabulary and customer text tokens
        criteria_keywords = INTENT_RESOLUTION_CRITERIA.get(predicted_intent, ["dm", "help", "order"])
        kw_hits = sum(1 for kw in criteria_keywords if kw in reply_lower)

        msg_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", msg_lower))
        reply_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", reply_lower))
        word_overlap = len(msg_words.intersection(reply_words))

        rel_score = 3.0
        if kw_hits >= 3 or word_overlap >= 3:
            rel_score = 5.0
        elif kw_hits >= 2 or word_overlap >= 1:
            rel_score = 4.0
        elif kw_hits == 1:
            rel_score = 3.0
        else:
            rel_score = 2.0

        # -------------------------------------------------------------
        # 2. Correctness Score (1-5)
        # -------------------------------------------------------------
        # Checks security guardrails and procedural validity
        cor_score = 5.0
        # Critical security check: Never ask user for raw password in public tweet
        if "send your password" in reply_lower or "post your password" in reply_lower:
            cor_score = 1.0
        # Financial checks: should advise checking via secure channel, not making unverified payouts
        elif predicted_intent == "refund_request" and "refunded your card right now" in reply_lower:
            cor_score = 2.0
        # Length sanity check: too short (< 20 chars) is inadequate
        elif len(generated_reply.strip()) < 25:
            cor_score = 2.0

        # -------------------------------------------------------------
        # 3. Grounding Score (1-5)
        # -------------------------------------------------------------
        # Assesses whether response is derived from retrieved precedent
        if not grounding_sources:
            gro_score = 2.0
        else:
            # Check lexical overlap with historical replies in retrieved cases
            hist_replies = " ".join([c.get("brand_reply_text", "").lower() for c in retrieved_cases])
            hist_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", hist_replies))
            overlap_count = len(reply_words.intersection(hist_words))

            if overlap_count >= 6:
                gro_score = 5.0
            elif overlap_count >= 3:
                gro_score = 4.0
            else:
                gro_score = 3.0

        # -------------------------------------------------------------
        # 4. Helpfulness Score (1-5)
        # -------------------------------------------------------------
        # Actionable next step indicator (DM link, help URL, instructions to provide details)
        help_markers = ["dm", "direct message", "link", "visit", "amzn.to", "reach out", "send us"]
        marker_hits = sum(1 for m in help_markers if m in reply_lower)

        if marker_hits >= 2 and len(generated_reply) > 50:
            hlp_score = 5.0
        elif marker_hits >= 1:
            hlp_score = 4.0
        else:
            hlp_score = 3.0

        # -------------------------------------------------------------
        # 5. Brand Consistency Score (1-5)
        # -------------------------------------------------------------
        # Politeness, greeting, and empathy check
        greetings = ["hi", "hello", "we're sorry", "apologize", "thank you", "thanks"]
        polite_hits = sum(1 for g in greetings if g in reply_lower)

        if polite_hits >= 2:
            brd_score = 5.0
        elif polite_hits >= 1:
            brd_score = 4.0
        else:
            brd_score = 3.0

        overall = round((rel_score + cor_score + gro_score + hlp_score + brd_score) / 5.0, 2)

        return {
            "relevance": rel_score,
            "correctness": cor_score,
            "grounding": gro_score,
            "helpfulness": hlp_score,
            "brand_consistency": brd_score,
            "overall_score": overall,
            "judge_backend": "rubric_heuristic",
            "notes": f"kw_hits={kw_hits}, overlap={word_overlap}, markers={marker_hits}"
        }

    def _score_llm(
        self,
        customer_message: str,
        predicted_intent: str,
        generated_reply: str,
        retrieved_cases: List[Dict[str, Any]],
        grounding_sources: List[str]
    ) -> Dict[str, Any]:
        """LLM-as-a-judge backend using external API if available."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            res = self._score_heuristic(customer_message, predicted_intent, generated_reply, retrieved_cases, grounding_sources)
            res["judge_backend"] = "rubric_heuristic_fallback"
            return res

        # If key is available, execute structured LLM rubric evaluation
        try:
            import requests
            prompt = (
                "You are an expert customer support quality auditor. Evaluate the following support reply on a 1-5 integer scale across:\n"
                "1. Relevance (1-5)\n2. Correctness (1-5)\n3. Grounding (1-5)\n4. Helpfulness (1-5)\n5. Brand Consistency (1-5)\n\n"
                f"Customer Query: {customer_message}\n"
                f"Intent: {predicted_intent}\n"
                f"Generated Reply: {generated_reply}\n\n"
                "Return scores in JSON format: {'relevance': X, 'correctness': X, 'grounding': X, 'helpfulness': X, 'brand_consistency': X}"
            )
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                import json
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                scores = [parsed["relevance"], parsed["correctness"], parsed["grounding"], parsed["helpfulness"], parsed["brand_consistency"]]
                parsed["overall_score"] = round(sum(scores) / len(scores), 2)
                parsed["judge_backend"] = "llm_openai"
                parsed["notes"] = "Evaluated via OpenAI gpt-3.5-turbo"
                return parsed
        except Exception as e:
            print(f"[llm_judge] LLM scoring failed ({e}). Reverting to rubric heuristic.")

        return self._score_heuristic(customer_message, predicted_intent, generated_reply, retrieved_cases, grounding_sources)

    def evaluate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        generated_reply: str,
        retrieved_cases: List[Dict[str, Any]],
        grounding_sources: List[str]
    ) -> Dict[str, Any]:
        """Main entry point for reply evaluation."""
        if self.use_llm:
            return self._score_llm(customer_message, predicted_intent, generated_reply, retrieved_cases, grounding_sources)
        return self._score_heuristic(customer_message, predicted_intent, generated_reply, retrieved_cases, grounding_sources)


if __name__ == "__main__":
    judge = SupportEvaluationJudge(use_llm=False)
    sample_eval = judge.evaluate_reply(
        customer_message="Where is my order #112-9281726? It was supposed to arrive yesterday!",
        predicted_intent="order_delivery_delay",
        generated_reply="Hi there, we're sorry your order #112-9281726 is delayed! Please send us a DM at amzn.to/help with your details so we can investigate.",
        retrieved_cases=[{"brand_reply_text": "Hi there, we're sorry your order is delayed! Please send us a DM at amzn.to/help with your details."}],
        grounding_sources=["100021"]
    )
    print("Judge Evaluation Scores:")
    for k, v in sample_eval.items():
        print(f"  {k}: {v}")
