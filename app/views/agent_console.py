"""
View: Autonomous Agent Console
Single-query interactive testing console with intent classification,
explainable safety gates, grounded response generation, and historical precedent evidence cards.
"""

import pandas as pd
import streamlit as st

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


def render_agent_console(pipeline, retrieval_mode: str = "hybrid"):
    st.markdown('<div class="main-header">Autonomous Agent Console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Test intent classification, grounded historical case retrieval, and explainable escalation safety controls in real time.</div>',
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

    col_act1, col_act2 = st.columns([1.5, 4.5])
    with col_act1:
        analyze_clicked = st.button("🚀 Process Message", type="primary", use_container_width=True)

    if analyze_clicked or user_message:
        with st.spinner("Analyzing message through support pipeline..."):
            result = pipeline.process_message(user_message, retrieval_mode=retrieval_mode)

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

            thresh = result.get("applied_threshold", 0.55)
            boost = result.get("precedent_boost_applied", False)
            risk = result.get("risk_tier", "STANDARD_RISK")
            boost_text = " (⚡ Grounding Boost)" if boost else ""
            st.caption(f"🛡️ Class Threshold Floor: `{thresh*100:.0f}%`{boost_text} | Risk: `{risk}`")

            with st.expander("📊 View All Intent Probabilities", expanded=False):
                prob_df = pd.DataFrame(
                    list(result["all_intent_probabilities"].items()),
                    columns=["Intent", "Probability"]
                ).sort_values(by="Probability", ascending=False)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

            st.markdown("#### 🔍 Historical Case Grounding")
            sim = result["top_retrieval_similarity"]
            st.metric(label="Top Precedent Grounding", value=f"{sim:.3f}")
            sources = ", ".join(result["grounding_sources"]) if result["grounding_sources"] else "None (Low Sim / Fallback)"
            st.markdown(f"**Grounded Source IDs:** `{sources}`")
            engine_tag = result.get("retrieval_mode", retrieval_mode).upper()
            st.caption(f"Generation: `{result['generation_mode']}` | Retrieval Engine: `{engine_tag}`")

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
            st.text_area("Raw Reply Text (Copyable):", value=result["generated_reply"], height=90, key="console_reply_copy")

        # Section 3: Historical Precedents (Evidence Cards)
        st.markdown("### 📚 Top-3 Historical Evidence Precedents")
        st.caption(f"Retrieved from AmazonHelp resolution archive using **{retrieval_mode.upper()}** mode.")

        cases = result.get("retrieved_cases", [])
        if cases:
            c_cols = st.columns(len(cases))
            for i, case in enumerate(cases):
                with c_cols[i]:
                    sim_score = case.get("similarity_score", 0.0)
                    sim_pct = f"{sim_score*100:.1f}%"
                    dense_score = case.get("dense_score", 0.0)
                    sparse_score = case.get("sparse_score", 0.0)
                    score_badge = f"Dense: {dense_score*100:.0f}% | Sparse: {sparse_score*100:.0f}%" if case.get("retrieval_mode") == "hybrid" else f"TF-IDF: {sparse_score*100:.0f}%"
                    st.markdown(
                        f"""
                        <div class="evidence-card">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                                <strong>Case #{i+1}</strong>
                                <span class="tag">Sim: {sim_pct}</span>
                            </div>
                            <div style="font-size: 0.72rem; color: #6366F1; margin-bottom: 0.3rem; font-weight: 500;">
                                {score_badge}
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
