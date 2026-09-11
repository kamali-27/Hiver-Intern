"""
End-to-End Customer Support Pipeline.

Single entry point connecting:
1. Intent Classifier (ML model)
2. Case Retriever (Vector index)
3. Response Generator (Grounded generation)
4. Escalation Engine (Interpretable safety logic)
"""

import os
import sys
import joblib
import numpy as np
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval import CaseRetriever, get_or_build_retriever
from src.response_generator import ResponseGenerator
from src.escalation_logic import EscalationEngine

FINAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_intent_classifier.pkl")


class CustomerSupportPipeline:
    """
    Unified production pipeline for processing incoming customer support messages.
    """
    def __init__(self, use_llm: Optional[bool] = None):
        self.model = self._load_classifier()
        self.retriever = get_or_build_retriever()
        self.generator = ResponseGenerator(use_llm=use_llm)
        self.escalation_engine = EscalationEngine()

    def _load_classifier(self):
        """Loads trained final intent classifier or triggers training if missing."""
        if not os.path.exists(FINAL_MODEL_PATH):
            print(f"[pipeline] Classifier not found at {FINAL_MODEL_PATH}. Training models now...")
            from src.train_intent_classifier import train_and_evaluate_all
            train_and_evaluate_all()
        return joblib.load(FINAL_MODEL_PATH)

    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Executes full pipeline: Classify -> Retrieve -> Generate -> Escalate.
        """
        clean_msg = (message or "").strip()

        # Handle empty/whitespace input
        if not clean_msg:
            return {
                "message": message,
                "predicted_intent": "general_inquiry",
                "intent_confidence": 0.0,
                "all_intent_probabilities": {},
                "retrieved_cases": [],
                "top_retrieval_similarity": 0.0,
                "generated_reply": "Hello! We didn't receive any message content. How can we help you today?",
                "grounding_sources": [],
                "generation_mode": "fallback_empty",
                "escalation_decision": "ESCALATE_TO_HUMAN",
                "escalation_reason": "Customer message was empty or invalid.",
                "escalation_flags": ["EMPTY_MESSAGE"]
            }

        # 1. Intent Classification
        probas = self.model.predict_proba([clean_msg])[0]
        classes = list(self.model.classes_)
        best_idx = int(np.argmax(probas))
        predicted_intent = classes[best_idx]
        confidence = float(probas[best_idx])

        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probas)}

        # 2. Case Retrieval
        retrieved_cases = self.retriever.retrieve(clean_msg, k=3)
        top_similarity = retrieved_cases[0]["similarity_score"] if retrieved_cases else 0.0

        # 3. Response Generation
        gen_result = self.generator.generate(clean_msg, predicted_intent, retrieved_cases)

        # 4. Escalation Evaluation
        esc_result = self.escalation_engine.evaluate(
            message=clean_msg,
            predicted_intent=predicted_intent,
            intent_confidence=confidence,
            top_retrieval_similarity=top_similarity
        )

        return {
            "message": clean_msg,
            "predicted_intent": predicted_intent,
            "intent_confidence": round(confidence, 4),
            "all_intent_probabilities": prob_dict,
            "retrieved_cases": retrieved_cases,
            "top_retrieval_similarity": round(top_similarity, 4),
            "generated_reply": gen_result["reply_text"],
            "grounding_sources": gen_result["grounding_sources"],
            "generation_mode": gen_result["generation_mode"],
            "escalation_decision": esc_result["decision"],
            "escalation_reason": esc_result["reason"],
            "escalation_flags": esc_result["flags"]
        }


# Singleton pipeline instance for reuse
_PIPELINE_INSTANCE = None

def get_pipeline(use_llm: Optional[bool] = None) -> CustomerSupportPipeline:
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = CustomerSupportPipeline(use_llm=use_llm)
    return _PIPELINE_INSTANCE


if __name__ == "__main__":
    pipeline = get_pipeline()
    sample = "Where is my package? Order #102-4491023 was supposed to arrive 2 days ago!"
    result = pipeline.process_message(sample)
    print("\n--- Pipeline Result ---")
    print(f"Input: {result['message']}")
    print(f"Intent: {result['predicted_intent']} (conf={result['intent_confidence']:.2f})")
    print(f"Top Sim: {result['top_retrieval_similarity']:.2f}")
    print(f"Decision: {result['escalation_decision']} ({result['escalation_reason']})")
    print(f"Reply: {result['generated_reply']}")
    print(f"Sources: {result['grounding_sources']}")
