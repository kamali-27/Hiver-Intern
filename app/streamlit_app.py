"""
Streamlit Web Application: AI Customer Support Agent
Single-page demo interface for testing intent classification, case retrieval,
grounded response generation, and safety escalation logic.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import get_pipeline

# Configure Streamlit page
st.set_page_config(
    page_title="AI Customer Support Agent | Autonomous Helpdesk",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished, executive-ready presentation
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    
    .badge-auto {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 0.55rem 1.1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    
    .badge-escalate {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 0.55rem 1.1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.2);
    }
    
    .reason-box {
        background-color: #F9FAFB;
        border-left: 4px solid #3B82F6;
        padding: 0.85rem 1.1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        font-size: 0.95rem;
        color: #1F2937;
    }
    
    .reason-box-escalate {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 0.85rem 1.1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        font-size: 0.95rem;
        color: #991B1B;
    }
    
    .response-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    
    .evidence-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
    }
    
    .tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample test cases for one-click demo
SAMPLE_QUERIES = [
    {
        "label": "📦 Delivery Delay (Routine)",
        "query": "Where is my package? Order #112-9281726 was supposed to arrive yesterday and tracking has not updated!",
    },
    {
        "label": "🚨 Angry Escalation / Legal (High Risk)",
        "query": "Your carrier threw my package into the mud and ruined my sister's wedding gift. I demand to speak with a supervisor immediately or I will contact my lawyer!",
    },
    {
        "label": "💰 Refund Request (Financial Action)",
        "query": "The product arrived completely shattered. I returned it to the locker 4 days ago. Where is my $120.00 refund?",
    },
    {
        "label": "💳 Billing Discrepancy",
        "query": "I noticed an unexpected duplicate charge of $14.99 on my credit card statement for Prime Video. Can you refund this?",
    },
    {
        "label": "🔐 Account Access Lockout",
        "query": "I am locked out of my Amazon account because 2-factor authentication is sending codes to my old disconnected phone number.",
    },
    {
        "label": "❌ Cancellation Request",
        "query": "Please cancel order #402-8827162 immediately! I placed it by accident 10 minutes ago.",
    },
    {
        "label": "❓ Ambiguous Query (Low Confidence)",
        "query": "I don't understand why this thing is like that, please check it.",
    }
]

# Initialize pipeline with caching
@st.cache_resource(show_spinner="Initializing AI Support Agent pipeline...")
def load_support_pipeline():
    return get_pipeline()

pipeline = load_support_pipeline()

# Load evaluation benchmarks for the sidebar
@st.cache_data
def load_benchmark_data():
    bench_path = os.path.join(PROJECT_ROOT, "eval", "results", "intent_comparison.csv")
    if os.path.exists(bench_path):
        return pd.read_csv(bench_path)
    return None

bench_df = load_benchmark_data()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/bot.png", width=64)
    st.markdown("### 🤖 Support Agent Console")
    st.markdown("**Version**: 1.0.0 (Hiver SDE Production Spec)")
    st.markdown("**Focus Brand**: `AmazonHelp` (Retail / E-Commerce)")
    st.markdown("---")
    
    st.markdown("#### 🎯 Intent Classification Benchmarks")
    if bench_df is not None:
        display_df = bench_df[["Model", "Accuracy", "F1 (Weighted)", "Recall (Weighted)"]].copy()
        for col in ["Accuracy", "F1 (Weighted)", "Recall (Weighted)"]:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.info("Run `python eval/run_evaluation.py` to populate benchmarks.")

    st.markdown("---")
    st.markdown("#### 🛡️ Escalation Safety Guardrails")
    st.markdown("""
    The system deterministically routes messages to humans when:
    - **Intent Confidence** < `0.55`
    - **High-Risk Intent** (`refund_request`, `complaint_escalation`, `cancellation_request`)
    - **Retrieval Grounding** < `0.25` similarity
    - **Trigger Keywords** (*supervisor*, *lawyer*, *fraud*, *bbb*, *dispute*)
    - **Extreme Frustration / Profanity**
    """)
    st.markdown("---")
    st.markdown("#### 📖 Documentation Links")
    st.markdown("- [Failure Analysis Report](docs/failure_analysis.md)")
    st.markdown("- [Why Headline Numbers Lie](docs/misleading_headline_number.md)")
    st.markdown("- [7-Day Improvement Roadmap](docs/one_week_improvements.md)")
    st.markdown("- [System Decision Log](docs/decision_log.md)")

# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-header">Autonomous Customer Support Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Data-driven intent classification, grounded historical case retrieval, and explainable escalation safety controls.</div>',
    unsafe_allow_html=True
)

# Quick preset selector
st.markdown("##### ⚡ Quick Select Test Scenarios:")
preset_cols = st.columns(4)
for idx, item in enumerate(SAMPLE_QUERIES):
    with preset_cols[idx % 4]:
        if st.button(item["label"], key=f"quick_{idx}", use_container_width=True):
            st.session_state["current_message_input"] = item["query"]
            st.rerun()

if "current_message_input" not in st.session_state:
    st.session_state["current_message_input"] = SAMPLE_QUERIES[0]["query"]

# Main text input
user_message = st.text_area(
    "Enter customer message:",
    key="current_message_input",
    height=100,
    placeholder="Type an inbound customer support inquiry..."
)

col_act1, col_act2, col_act3 = st.columns([1.5, 2, 2])
with col_act1:
    analyze_clicked = st.button("🚀 Process Message", type="primary", use_container_width=True)

if analyze_clicked or user_message:
    with st.spinner("Analyzing message through support pipeline..."):
        result = pipeline.process_message(user_message)
    
    st.markdown("---")
    
    # Section 1: Decision Banner & Explainable Escalation
    col_dec1, col_dec2 = st.columns([1.2, 2.8])
    with col_dec1:
        if result["escalation_decision"] == "AUTO_HANDLE":
            st.markdown(
                '<div class="badge-auto">✅ AUTO-HANDLE</div>',
                unsafe_allow_html=True
            )
            st.caption("Autonomous AI resolution approved")
        else:
            st.markdown(
                '<div class="badge-escalate">🚨 ESCALATE TO HUMAN</div>',
                unsafe_allow_html=True
            )
            st.caption("Routed to Human Support Specialist")
            
    with col_dec2:
        reason_class = "reason-box" if result["escalation_decision"] == "AUTO_HANDLE" else "reason-box-escalate"
        st.markdown(
            f'<div class="{reason_class}"><strong>Reason:</strong> {result["escalation_reason"]}</div>',
            unsafe_allow_html=True
        )
        if result.get("escalation_flags"):
            flag_badges = " ".join([f'<span class="tag">{f}</span>' for f in result["escalation_flags"]])
            st.markdown(f"**Triggered Safety Flags:** {flag_badges}", unsafe_allow_html=True)

    # Section 2: Intent Classification & Generation
    st.markdown("### 📝 Analysis & Resolution")
    col_res1, col_res2 = st.columns([1.3, 1.7])
    
    with col_res1:
        st.markdown("#### 🎯 Intent Detection")
        intent_name = result["predicted_intent"].replace("_", " ").title()
        conf = result["intent_confidence"]
        
        st.metric(label="Predicted Intent", value=intent_name, delta=f"{conf*100:.1f}% Confidence")
        st.progress(conf)
        
        with st.expander("📊 View All Intent Probabilities", expanded=False):
            prob_df = pd.DataFrame(
                list(result["all_intent_probabilities"].items()),
                columns=["Intent", "Probability"]
            ).sort_values(by="Probability", ascending=False)
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
            
        st.markdown("#### 🔍 Historical Case Grounding")
        sim = result["top_retrieval_similarity"]
        st.metric(label="Top Precedent Similarity", value=f"{sim:.3f}")
        sources = ", ".join(result["grounding_sources"]) if result["grounding_sources"] else "None (Low Sim / Fallback)"
        st.markdown(f"**Grounded Source IDs:** `{sources}`")
        st.caption(f"Generation Engine: `{result['generation_mode']}`")

    with col_res2:
        st.markdown("#### 💬 Generated Brand Response")
        st.markdown(
            f"""
            <div class="response-card">
                <div style="color: #6B7280; font-size: 0.8rem; margin-bottom: 0.5rem; text-transform: uppercase; font-weight: 600;">
                    Proposed Outbound Reply
                </div>
                <div style="font-size: 1.05rem; color: #111827; line-height: 1.6;">
                    {result["generated_reply"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.text_area("Raw Reply Text (Copyable):", value=result["generated_reply"], height=90)
        
    # Section 3: Historical Precedents (Evidence Cards)
    st.markdown("### 📚 Top-3 Historical Evidence Precedents")
    st.caption("Retrieved from historical Amazon customer support resolution archive via TF-IDF cosine similarity vector index.")
    
    cases = result.get("retrieved_cases", [])
    if cases:
        c_cols = st.columns(len(cases))
        for i, case in enumerate(cases):
            with c_cols[i]:
                sim_score = case.get("similarity_score", 0.0)
                sim_pct = f"{sim_score*100:.1f}%"
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                            <strong>Case #{i+1}</strong>
                            <span class="tag">Sim: {sim_pct}</span>
                        </div>
                        <div style="font-size: 0.75rem; color: #64748B; margin-bottom: 0.4rem;">
                            ID: <code>{case.get('conversation_id')}</code> | Intent: <code>{case.get('intent')}</code>
                        </div>
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Customer:</strong><br/>
                            <span style="color: #334155;">"{case.get('customer_text', '')[:140]}..."</span>
                        </div>
                        <div>
                            <strong>Brand Reply:</strong><br/>
                            <span style="color: #1E293B;">"{case.get('brand_reply_text', '')[:140]}..."</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("No historical precedent found for this query.")
