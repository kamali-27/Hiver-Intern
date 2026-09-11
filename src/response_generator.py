"""
Response Generation Module.

Generates grounded support responses given:
- Incoming customer message
- Predicted intent
- Top-k retrieved historical cases

Supports two execution paths:
1. Default Path (`USE_LLM=false`): Template + extraction-based generator that selects
   and adapts historical brand resolutions, extracts order/account references,
   and ensures high grounding without hallucination or API key requirements.
2. Optional Path (`USE_LLM=true`): Local/API LLM call prompted with retrieved historical
   cases as grounding context (gracefully falls back to default if no key configured).
"""

import os
import re
from typing import List, Dict, Any, Optional

# Intent-specific resolution guidance templates (used when adapting retrieved resolutions)
INTENT_FALLBACK_TEMPLATES = {
    "order_delivery_delay": (
        "Hello! We apologize for the delay with {order_ref}. "
        "Please check your tracking details at amzn.to/track, and if the package has not arrived, "
        "send us a direct message with your order number and delivery postal code so we can assist."
    ),
    "refund_request": (
        "Hi there, we apologize for the trouble with your return/refund for {order_ref}. "
        "Standard refunds take 3-5 business days once received. Please DM us your order and return details "
        "so our billing specialists can verify the status."
    ),
    "account_access": (
        "Hello! For your account security, please never post passwords or verification codes publicly. "
        "Please visit amzn.to/account-recovery to begin identity verification or send us a DM for secure assistance."
    ),
    "billing_issue": (
        "Hi, we'd be happy to investigate this charge for you. "
        "Please reach out via direct message with the transaction date, amount, and last 4 digits of the card charged "
        "so we can review your account securely."
    ),
    "technical_bug": (
        "We're sorry you're experiencing technical difficulties! "
        "Please try clearing your app cache or testing in an incognito browser window. If the issue persists, "
        "please DM us your device model and app version so our technical team can look into it."
    ),
    "complaint_escalation": (
        "We are truly sorry for your frustrating experience; this is definitely not the standard of service we strive for. "
        "We want to make this right immediately. Please send us a direct message with your contact phone number and "
        "order details so a supervisor can directly review and assist."
    ),
    "positive_feedback": (
        "Thank you so much for the kind feedback! We are thrilled to hear you had a great experience, "
        "and we will be sure to share your compliments with our team. Have a wonderful day!"
    ),
    "general_inquiry": (
        "Hi! Thanks for reaching out. You can find detailed information on our policies and services at amzn.to/help. "
        "If you have specific details regarding {order_ref}, please feel free to send us a direct message!"
    )
}


class ResponseGenerator:
    """
    Dual-mode Grounded Response Generator.
    """
    def __init__(self, use_llm: Optional[bool] = None):
        if use_llm is None:
            env_val = os.getenv("USE_LLM", "false").lower()
            self.use_llm = env_val in ("true", "1", "yes")
        else:
            self.use_llm = use_llm

    def _extract_slots(self, message: str) -> Dict[str, str]:
        """Extracts customer-provided references such as order numbers or amounts."""
        slots = {}

        # Extract order IDs (e.g. 112-9213812-1923812 or #123456)
        order_match = re.search(r"\b(\d{3}-\d{7}-\d{7})\b", message)
        if not order_match:
            order_match = re.search(r"#\s*(\d{4,10})\b", message)

        if order_match:
            slots["order_ref"] = f"order #{order_match.group(1)}"
        else:
            slots["order_ref"] = "your order"

        # Extract dollar amounts
        amt_match = re.search(r"\$(\d+(?:\.\d{2})?)", message)
        if amt_match:
            slots["amount"] = f"${amt_match.group(1)}"
        else:
            slots["amount"] = "the charge"

        return slots

    def _generate_template_grounded(
        self,
        message: str,
        predicted_intent: str,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Template + Extraction path: Selects and adapts top historical brand reply.
        """
        slots = self._extract_slots(message)
        grounding_sources = [case["conversation_id"] for case in retrieved_cases]

        top_case = retrieved_cases[0] if retrieved_cases else None
        top_similarity = top_case["similarity_score"] if top_case else 0.0

        # If we have a reasonably high similarity historical match (>= 0.22)
        if top_case and top_similarity >= 0.22:
            base_reply = top_case["brand_reply_text"]

            # Adapt order references in historical reply to customer's actual order reference
            adapted = re.sub(r"#?\b\d{3}-\d{7}-\d{7}\b", slots["order_ref"], base_reply)
            adapted = re.sub(r"#\d{4,10}\b", slots["order_ref"], adapted)

            # Adapt currency amounts
            adapted = re.sub(r"\$\d+(?:\.\d{2})?", slots["amount"], adapted)

            # Clean up duplicate phrase artifacts resulting from slot substitution
            adapted = re.sub(r"\border\s+order\b", "order", adapted, flags=re.IGNORECASE)
            adapted = re.sub(r"\border\s+your\s+order\b", "your order", adapted, flags=re.IGNORECASE)

            # Ensure customer feels acknowledged
            final_reply = adapted
        else:
            # If similarity is lower, fall back to intent-tailored resolution template
            template = INTENT_FALLBACK_TEMPLATES.get(
                predicted_intent,
                INTENT_FALLBACK_TEMPLATES["general_inquiry"]
            )
            final_reply = template.format(**slots)

        return {
            "reply_text": final_reply,
            "grounding_sources": grounding_sources,
            "generation_mode": "template_grounded",
            "primary_source_similarity": top_similarity
        }

    def _generate_llm(
        self,
        message: str,
        predicted_intent: str,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optional LLM path: Prompts an LLM with retrieved cases as grounding evidence.
        Falls back to template_grounded if no API key is set.
        """
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("[response_generator] USE_LLM=true requested but no API key configured. Falling back to template_grounded.")
            res = self._generate_template_grounded(message, predicted_intent, retrieved_cases)
            res["generation_mode"] = "template_grounded_fallback"
            return res

        # If API key is available, execute LLM call
        try:
            grounding_evidence = "\n".join([
                f"- Case #{c['conversation_id']} (intent={c['intent']}, score={c['similarity_score']:.2f}):\n"
                f"  Customer: {c['customer_text']}\n"
                f"  Brand Reply: {c['brand_reply_text']}"
                for c in retrieved_cases
            ])

            system_prompt = (
                "You are an empathetic, concise, and professional customer support assistant for AmazonHelp. "
                "You must strictly ground your response in the historical brand resolutions provided below. "
                "Never invent policies or claim actions you cannot take. Keep the answer under 3 sentences."
            )
            user_prompt = (
                f"Customer Query: {message}\n"
                f"Classified Intent: {predicted_intent}\n\n"
                f"Historical Grounding Cases:\n{grounding_evidence}\n\n"
                f"Generate the brand response:"
            )

            # Placeholder for actual API client call when key exists
            import requests
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 150
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                reply_text = data["choices"][0]["message"]["content"].strip()
                return {
                    "reply_text": reply_text,
                    "grounding_sources": [c["conversation_id"] for c in retrieved_cases],
                    "generation_mode": "llm_openai",
                    "primary_source_similarity": retrieved_cases[0]["similarity_score"] if retrieved_cases else 0.0
                }
            else:
                print(f"[response_generator] API returned {resp.status_code}. Using template fallback.")
        except Exception as e:
            print(f"[response_generator] LLM call failed ({e}). Using template fallback.")

        return self._generate_template_grounded(message, predicted_intent, retrieved_cases)

    def generate(
        self,
        message: str,
        predicted_intent: str,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Main generation entry point."""
        if self.use_llm:
            return self._generate_llm(message, predicted_intent, retrieved_cases)
        return self._generate_template_grounded(message, predicted_intent, retrieved_cases)


if __name__ == "__main__":
    generator = ResponseGenerator(use_llm=False)
    dummy_cases = [
        {
            "conversation_id": "100021",
            "customer_text": "Where is my package #102-3921932? It is delayed!",
            "brand_reply_text": "Hi there, we're sorry your package #102-3921932 is delayed! Please send us a DM at amzn.to/help with your details so we can investigate.",
            "intent": "order_delivery_delay",
            "similarity_score": 0.78
        }
    ]
    res = generator.generate(
        message="My shipment order #405-1122334 is still missing after 3 days.",
        predicted_intent="order_delivery_delay",
        retrieved_cases=dummy_cases
    )
    print("Generated Reply:\n", res["reply_text"])
    print("Sources:", res["grounding_sources"])
    print("Mode:", res["generation_mode"])
