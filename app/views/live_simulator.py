"""
View: Live Customer Support Simulator
Interactive multi-turn simulation sandbox with pre-built customer persona stress tests,
cumulative frustration tracking, entity extraction, and turn-by-turn safety audits.
"""

import time
import pandas as pd
import streamlit as st

from src.simulator_engine import (
    SimulatorEngine,
    STRESS_SCENARIOS,
    SimulatorState
)


def render_live_simulator(pipeline, retrieval_mode: str = "hybrid"):
    st.markdown('<div class="main-header">⚡ Live Support Dialog Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Simulate realistic multi-turn customer dialogues, stress-test safety escalation thresholds, and monitor stateful entity tracking.</div>',
        unsafe_allow_html=True
    )

    if "simulator_engine" not in st.session_state:
        st.session_state["simulator_engine"] = SimulatorEngine(pipeline=pipeline)
    sim_engine: SimulatorEngine = st.session_state["simulator_engine"]

    # Scenario Selection
    col_sc1, col_sc2 = st.columns([2.5, 1.5])
    with col_sc1:
        scenario_keys = list(STRESS_SCENARIOS.keys())
        scenario_labels = [STRESS_SCENARIOS[k]["title"] for k in scenario_keys]
        selected_idx = st.selectbox(
            "Select Customer Stress-Test Scenario:",
            range(len(scenario_keys)),
            format_func=lambda i: scenario_labels[i],
            key="sim_scenario_select"
        )
        selected_key = scenario_keys[selected_idx]
        scenario_info = STRESS_SCENARIOS[selected_key]

    with col_sc2:
        st.markdown(f"**Customer:** `{scenario_info['customer_name']}`")
        st.caption(scenario_info["description"])

    # Initialize or reset session state
    if (
        "active_sim_state" not in st.session_state
        or st.session_state.get("current_sim_key") != selected_key
    ):
        st.session_state["active_sim_state"] = sim_engine.create_session(selected_key)
        st.session_state["current_sim_key"] = selected_key
        st.session_state["sim_turn_pointer"] = 0

    state: SimulatorState = st.session_state["active_sim_state"]
    total_scripted_turns = len(scenario_info["turns"])
    current_turn_idx = st.session_state.get("sim_turn_pointer", 0)

    # Simulation Action Bar
    st.markdown("---")
    act_col1, act_col2, act_col3, act_col4 = st.columns([1.5, 1.5, 1.5, 2.5])

    with act_col1:
        can_advance = current_turn_idx < total_scripted_turns
        if st.button("▶️ Advance Next Turn", disabled=not can_advance, type="primary", use_container_width=True):
            next_msg = scenario_info["turns"][current_turn_idx]
            sim_engine.process_turn(state, next_msg, retrieval_mode=retrieval_mode)
            st.session_state["sim_turn_pointer"] = current_turn_idx + 1
            st.rerun()

    with act_col2:
        if st.button("⏩ Run All Turns", disabled=not can_advance, use_container_width=True):
            while st.session_state["sim_turn_pointer"] < total_scripted_turns:
                idx = st.session_state["sim_turn_pointer"]
                msg = scenario_info["turns"][idx]
                sim_engine.process_turn(state, msg, retrieval_mode=retrieval_mode)
                st.session_state["sim_turn_pointer"] += 1
            st.rerun()

    with act_col3:
        if st.button("🔄 Reset Simulation", use_container_width=True):
            st.session_state["active_sim_state"] = sim_engine.create_session(selected_key)
            st.session_state["sim_turn_pointer"] = 0
            st.rerun()

    with act_col4:
        st.caption(f"Scripted Progress: **Turn {len(state.turns)} / {total_scripted_turns}**")

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Conversation ID", value=state.conversation_id)
    with m_col2:
        status_color = "🟢" if state.current_status == "AUTO_RESOLVED" else ("🔴" if state.current_status == "ESCALATED_TO_HUMAN" else "🟡")
        st.metric(label="Session Routing Status", value=f"{status_color} {state.current_status.replace('_', ' ')}")
    with m_col3:
        frust = state.cumulative_frustration
        delta_label = "Low" if frust < 0.3 else ("Moderate" if frust < 0.6 else "Hostile")
        st.metric(label="Cumulative Frustration", value=f"{frust*100:.0f}%", delta=delta_label)
        st.progress(frust)
    with m_col4:
        entities_count = len(state.extracted_entities)
        st.metric(label="Entities Captured", value=f"{entities_count} items")

    # Side-by-side layout: Chat Timeline on Left, Context & Audit Ledger on Right
    st.markdown("---")
    chat_col, info_col = st.columns([1.8, 1.2])

    with chat_col:
        st.markdown("### 💬 Conversation Thread")
        if not state.turns:
            st.info(
                "Click **'▶️ Advance Next Turn'** or type a custom message below to begin the simulation."
            )
        else:
            for turn in state.turns:
                # Customer Bubble
                st.markdown(
                    f"""
                    <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 12px 12px 12px 0; padding: 0.9rem; margin-bottom: 0.6rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #1E40AF; margin-bottom: 0.3rem;">
                            <strong>👤 {state.customer_name} (Turn #{turn.turn_index})</strong>
                            <span>Sentiment: {turn.sentiment_score:+.2f} ({turn.frustration_level})</span>
                        </div>
                        <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.5;">
                            {turn.customer_message}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Agent Bubble
                badge = (
                    '<span style="background: #10B981; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.72rem;">AUTO-HANDLE</span>'
                    if turn.escalation_decision == "AUTO_HANDLE"
                    else '<span style="background: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.72rem;">ESCALATED TO HUMAN</span>'
                )

                st.markdown(
                    f"""
                    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px 12px 0 12px; padding: 0.9rem; margin-bottom: 1.2rem; margin-left: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #475569; margin-bottom: 0.4rem;">
                            <strong>🤖 AmazonHelp Support Agent</strong>
                            <div>{badge}</div>
                        </div>
                        <div style="color: #0F172A; font-size: 0.95rem; line-height: 1.5; margin-bottom: 0.5rem;">
                            {turn.agent_reply}
                        </div>
                        <div style="font-size: 0.75rem; color: #64748B; border-top: 1px solid #F1F5F9; padding-top: 0.4rem;">
                            🎯 Intent: <code>{turn.predicted_intent}</code> ({turn.intent_confidence*100:.0f}%) | 
                            🔍 Sim: <code>{turn.similarity_score:.3f}</code> | 
                            ⏱️ Latency: <code>{turn.latency_ms} ms</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Free-form customer input
        st.markdown("##### ✍️ Send Custom Inbound Customer Message:")
        custom_input_col1, custom_input_col2 = st.columns([4, 1])
        with custom_input_col1:
            custom_msg = st.text_input(
                "Custom customer reply",
                placeholder="Type your own customer reply to test live reaction...",
                label_visibility="collapsed",
                key="custom_sim_input"
            )
        with custom_input_col2:
            send_clicked = st.button("Send 💬", use_container_width=True)

        if send_clicked and custom_msg.strip():
            sim_engine.process_turn(state, custom_msg.strip(), retrieval_mode=retrieval_mode)
            st.rerun()

    with info_col:
        st.markdown("### 📋 Captured Session Entities")
        if state.extracted_entities:
            for k, v in state.extracted_entities.items():
                st.markdown(f"- **{k.replace('_', ' ').title()}:** `{v}`")
        else:
            st.caption("No specific entities (order IDs, tracking #, amounts) detected yet.")

        st.markdown("---")
        st.markdown("### 🛡️ Turn-by-Turn Safety Audit")
        if state.turns:
            audit_data = []
            for t in state.turns:
                audit_data.append({
                    "Turn": f"#{t.turn_index}",
                    "Intent": t.predicted_intent,
                    "Conf": f"{t.intent_confidence*100:.0f}%",
                    "Decision": t.escalation_decision,
                    "Flags": ", ".join(t.escalation_flags) if t.escalation_flags else "None"
                })
            audit_df = pd.DataFrame(audit_data)
            st.dataframe(audit_df, hide_index=True, use_container_width=True)
            
            last_turn = state.turns[-1]
            st.markdown(f"**Latest Routing Reason:**")
            st.caption(last_turn.escalation_reason)
        else:
            st.caption("Audit trail will log each turn's safety flags and decisions.")
