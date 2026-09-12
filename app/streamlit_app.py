"""
Streamlit Web Application: Enterprise AI Customer Support Suite (Hiver/Zendesk Style)
Multi-view support intelligence platform featuring:
1. 🛡️ Autonomous Agent Console
2. ⚡ Live Support Dialog Simulator
3. 📊 Brand Intelligence & Operations Dashboard
4. 📚 RAG Knowledge Base & Precedent Search
5. 📥 Enterprise Support Ticket Inbox
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
from app.views.agent_console import render_agent_console
from app.views.live_simulator import render_live_simulator
from app.views.brand_analysis import render_brand_analysis
from app.views.rag_search import render_rag_search
from app.views.ticket_inbox import render_ticket_inbox

# Configure Streamlit page
st.set_page_config(
    page_title="Autonomous AI Customer Support Suite | Hiver SDE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
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


# ----------------- SIDEBAR NAVIGATION & CONFIG -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/bot.png", width=60)
    st.markdown("### 🛡️ Helpdesk Suite")
    st.caption("Autonomous AI Support Platform (Hiver SDE Style)")
    st.markdown("---")

    # Module Navigation
    st.markdown("#### 🧭 Platform Modules")
    selected_view = st.radio(
        "Select Workspace View:",
        options=[
            "🛡️ Agent Console",
            "⚡ Live Simulator",
            "📊 Brand Analysis",
            "📚 RAG Knowledge Search",
            "📥 Ticket Inbox"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("#### ⚡ Case Retrieval Engine")
    retrieval_choice = st.radio(
        "Select Retrieval Mode:",
        options=["Hybrid (Dense + Sparse RRF)", "Sparse (TF-IDF Only)"],
        index=0,
        help="Switch between Hybrid bi-encoder semantic search (all-MiniLM-L6-v2 + RRF) and traditional lexical TF-IDF."
    )
    selected_retrieval_mode = "hybrid" if "Hybrid" in retrieval_choice else "sparse"

    st.markdown("---")
    st.markdown("#### 🎯 Classification Benchmarks")
    if bench_df is not None:
        display_df = bench_df[["Model", "Accuracy", "F1 (Weighted)"]].copy()
        for col in ["Accuracy", "F1 (Weighted)"]:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.info("Run `python eval/run_evaluation.py` to populate benchmarks.")

    st.markdown("---")
    st.markdown("#### 📖 In-Depth Docs")
    st.markdown("- [Failure Analysis](docs/failure_analysis.md)")
    st.markdown("- [Misleading Headline Numbers](docs/misleading_headline_number.md)")
    st.markdown("- [7-Day Improvement Roadmap](docs/one_week_improvements.md)")
    st.markdown("- [Technical Decision Log](docs/decision_log.md)")


# ----------------- ROUTE VIEW -----------------
if "Agent Console" in selected_view:
    render_agent_console(pipeline, retrieval_mode=selected_retrieval_mode)

elif "Live Simulator" in selected_view:
    render_live_simulator(pipeline, retrieval_mode=selected_retrieval_mode)

elif "Brand Analysis" in selected_view:
    render_brand_analysis()

elif "RAG Knowledge Search" in selected_view:
    render_rag_search(pipeline, retrieval_mode=selected_retrieval_mode)

elif "Ticket Inbox" in selected_view:
    render_ticket_inbox(pipeline)
