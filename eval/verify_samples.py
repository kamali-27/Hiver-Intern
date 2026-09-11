import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import get_pipeline

def main():
    pipeline = get_pipeline()
    samples = [
        "Where is my package? Order #112-9281726 is late and tracking has not updated!",
        "Your courier destroyed my package and customer service hung up. I want to speak to a supervisor immediately or I will contact my lawyer!",
        "I returned order #402-8827162 5 days ago and still have not received my $120 refund.",
        "Can you help me cancel my order before it ships out?"
    ]
    
    print("\n" + "="*80)
    print("PIPELINE SAMPLE VERIFICATION TEST")
    print("="*80)
    for i, msg in enumerate(samples, 1):
        res = pipeline.process_message(msg)
        print(f"\n[Scenario #{i}] Input: {res['message']}")
        print(f"  Decision:    {res['escalation_decision']} ({res['escalation_reason']})")
        print(f"  Intent:      {res['predicted_intent']} (confidence: {res['intent_confidence']:.2f})")
        print(f"  Top Sim:     {res['top_retrieval_similarity']:.3f}")
        print(f"  Reply:       {res['generated_reply']}")
        print(f"  Sources:     {res['grounding_sources']}")
        print(f"  Flags:       {res['escalation_flags']}")
    print("\n" + "="*80)
    print("ALL SAMPLE SCENARIOS EXECUTED SUCCESSFULLY")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
