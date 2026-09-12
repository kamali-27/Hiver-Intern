"""
View: Enterprise Helpdesk Ticket Inbox
Production-grade support ticket triage inbox (Hiver/Zendesk style) with
queue filtering, ticket detail drawer, editable AI draft responses, and agent triage actions.
"""

import os
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FULL_OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "eval", "results", "full_system_outputs.csv")


def load_inbox_tickets() -> pd.DataFrame:
    """Loads evaluated tickets with metadata and decision labels."""
    if os.path.exists(FULL_OUTPUTS_PATH):
        df = pd.read_csv(FULL_OUTPUTS_PATH)
    else:
        # Fallback realistic tickets
        data = [
            ("GOLD_001", "Where is my package? Order #112-9281726 was supposed to arrive yesterday!", "order_delivery_delay", 0.87, 0.72, "AUTO_HANDLE", "", "So sorry for the delay on your order! Please DM us your order details..."),
            ("GOLD_002", "Your carrier stole my package and lied to me! Let me speak to a supervisor now!", "complaint_escalation", 0.92, 0.42, "ESCALATE_TO_HUMAN", "EXPLICIT_HUMAN_REQUEST;HOSTILE", "We sincerely apologize for this distressing experience..."),
            ("GOLD_003", "Where is my $120.00 refund for return #259-4612365? Dropped it off 4 days ago.", "refund_request", 0.88, 0.45, "ESCALATE_TO_HUMAN", "HIGH_RISK_INTENT;FINANCIAL_DISPUTE_GUARDRAIL", "We want to ensure your refund is handled promptly..."),
            ("GOLD_004", "I noticed a duplicate charge of $14.99 on my Visa statement for Prime Video.", "billing_issue", 0.75, 0.48, "ESCALATE_TO_HUMAN", "FINANCIAL_DISPUTE_GUARDRAIL", "Sorry for the unexpected duplicate charge! Please DM us...")
        ]
        df = pd.DataFrame(data, columns=[
            "message_id", "message", "predicted_intent", "intent_confidence",
            "top_similarity", "system_decision", "escalation_flags", "generated_reply"
        ])

    # Ensure required columns exist
    if "ticket_status" not in df.columns:
        df["ticket_status"] = df["system_decision"].apply(
            lambda d: "AUTO_RESOLVED" if d == "AUTO_HANDLE" else "PENDING_SPECIALIST"
        )
    return df


def render_ticket_inbox(pipeline):
    st.markdown('<div class="main-header">📥 Enterprise Support Ticket Inbox</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Manage, triage, and review inbound customer inquiries across autonomous and human-escalated queues.</div>',
        unsafe_allow_html=True
    )

    if "ticket_inbox_df" not in st.session_state:
        st.session_state["ticket_inbox_df"] = load_inbox_tickets()
    if "ticket_notes" not in st.session_state:
        st.session_state["ticket_notes"] = {}
    if "ticket_actions" not in st.session_state:
        st.session_state["ticket_actions"] = {}

    df_tickets: pd.DataFrame = st.session_state["ticket_inbox_df"]

    # Queue Selector
    auto_count = int((df_tickets["system_decision"] == "AUTO_HANDLE").sum())
    human_count = int((df_tickets["system_decision"] == "ESCALATE_TO_HUMAN").sum())
    urgent_mask = (
        df_tickets["predicted_intent"].isin(["refund_request", "complaint_escalation", "billing_issue"])
        | df_tickets["escalation_flags"].astype(str).str.contains("FINANCIAL|LEGAL|EXPLICIT")
    )
    urgent_count = int(urgent_mask.sum())

    st.markdown("##### 🗂️ Support Queues:")
    queue_selection = st.radio(
        "Filter by Queue:",
        options=[
            f"📥 All Tickets ({len(df_tickets)})",
            f"⚡ Auto-Resolved ({auto_count})",
            f"🚨 Human Specialist Queue ({human_count})",
            f"⏳ High Urgency Financial & Legal ({urgent_count})"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    if "Auto-Resolved" in queue_selection:
        filtered_df = df_tickets[df_tickets["system_decision"] == "AUTO_HANDLE"]
    elif "Human Specialist Queue" in queue_selection:
        filtered_df = df_tickets[df_tickets["system_decision"] == "ESCALATE_TO_HUMAN"]
    elif "High Urgency" in queue_selection:
        filtered_df = df_tickets[urgent_mask]
    else:
        filtered_df = df_tickets

    # Search and Filter Toolbar
    st.markdown("---")
    f_col1, f_col2, f_col3 = st.columns([2.5, 1.5, 1.5])
    with f_col1:
        keyword_search = st.text_input("🔍 Search tickets by text or ID:", placeholder="Type to search tickets...", key="inbox_kw_search")
    with f_col2:
        all_intents = ["All Intents"] + sorted(list(df_tickets["predicted_intent"].dropna().unique()))
        selected_intent = st.selectbox("Filter Intent:", all_intents, key="inbox_intent_filter")
    with f_col3:
        all_statuses = ["All Statuses", "AUTO_RESOLVED", "PENDING_SPECIALIST", "CLOSED", "ESCALATED_TIER_2"]
        selected_status = st.selectbox("Filter Status:", all_statuses, key="inbox_status_filter")

    # Apply filters
    display_df = filtered_df.copy()
    if keyword_search.strip():
        kw = keyword_search.strip().lower()
        display_df = display_df[
            display_df["message"].str.lower().str.contains(kw)
            | display_df["message_id"].str.lower().str.contains(kw)
        ]
    if selected_intent != "All Intents":
        display_df = display_df[display_df["predicted_intent"] == selected_intent]
    if selected_status != "All Statuses":
        display_df = display_df[display_df["ticket_status"] == selected_status]

    # Layout: Ticket List on Left, Ticket Detail Workspace on Right
    st.markdown("---")
    list_col, detail_col = st.columns([1.6, 2.4])

    with list_col:
        st.markdown(f"##### Showing {len(display_df)} Tickets")
        if display_df.empty:
            st.info("No tickets match current filters.")
            selected_ticket_id = None
        else:
            ticket_options = list(display_df["message_id"])
            selected_ticket_id = st.radio(
                "Select a ticket to review:",
                ticket_options,
                format_func=lambda tid: f"{tid} — {display_df.loc[display_df['message_id'] == tid, 'predicted_intent'].values[0]} ({display_df.loc[display_df['message_id'] == tid, 'system_decision'].values[0]})",
                key="inbox_selected_radio"
            )

    with detail_col:
        if not selected_ticket_id:
            st.info("Select a ticket on the left to open the triage workspace.")
        else:
            row = df_tickets[df_tickets["message_id"] == selected_ticket_id].iloc[0]
            current_status = st.session_state["ticket_actions"].get(selected_ticket_id, row.get("ticket_status", "ACTIVE"))

            # Ticket Header Banner
            status_badge = (
                '<span style="background: #10B981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;">⚡ AUTO-RESOLVED</span>'
                if current_status == "AUTO_RESOLVED"
                else ('<span style="background: #EF4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;">🚨 PENDING HUMAN REVIEW</span>'
                      if current_status == "PENDING_SPECIALIST"
                      else f'<span style="background: #6366F1; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;">{current_status}</span>')
            )

            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.85rem 1.1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <div>
                        <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">Ticket {row['message_id']}</span>
                        <span class="tag" style="margin-left: 0.5rem;">{row['predicted_intent']}</span>
                    </div>
                    <div>{status_badge}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Customer Message Panel
            st.markdown("##### 👤 Inbound Customer Message:")
            st.markdown(
                f"""
                <div style="background: #EFF6FF; border-left: 4px solid #3B82F6; padding: 0.9rem 1.1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem; color: #1E293B; font-size: 0.98rem; line-height: 1.5;">
                    "{row['message']}"
                </div>
                """,
                unsafe_allow_html=True
            )

            # AI Triage & Routing Explanations
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("Intent Confidence", f"{float(row.get('intent_confidence', 0.8)):.1%}")
            with col_t2:
                st.metric("Grounding Precedent", f"{float(row.get('top_similarity', 0.5)):.3f}")
            with col_t3:
                st.metric("System Routing", row.get("system_decision", "AUTO_HANDLE"))

            flags = str(row.get("escalation_flags", ""))
            if flags and flags != "nan":
                flag_badges = " ".join([f'<span class="tag">{f}</span>' for f in flags.split(";") if f.strip()])
                st.markdown(f"**Triggered Safety Guardrails:** {flag_badges}", unsafe_allow_html=True)

            reason = str(row.get("escalation_reason", ""))
            if reason and reason != "nan":
                st.caption(f"**Safety Rationale:** {reason}")

            st.markdown("---")

            # Outbound Response Editor
            st.markdown("##### 💬 Outbound Support Reply (AI Grounded Draft):")
            default_reply = str(row.get("generated_reply", "Hello! How can we assist you today?"))
            agent_reply_text = st.text_area(
                "Edit Outbound Reply:",
                value=default_reply,
                height=110,
                key=f"reply_edit_{selected_ticket_id}"
            )

            # Action Buttons
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                if st.button("✅ Approve AI Reply", key=f"approve_{selected_ticket_id}", type="primary", use_container_width=True):
                    st.session_state["ticket_actions"][selected_ticket_id] = "AUTO_RESOLVED"
                    st.success(f"Ticket {selected_ticket_id} approved and resolved autonomously!")
                    st.rerun()

            with b_col2:
                if st.button("✍️ Send Custom", key=f"custom_{selected_ticket_id}", use_container_width=True):
                    st.session_state["ticket_actions"][selected_ticket_id] = "AGENT_INTERVENED"
                    st.success(f"Custom reply dispatched to customer!")
                    st.rerun()

            with b_col3:
                if st.button("👨‍💼 Escalate Tier-2", key=f"tier2_{selected_ticket_id}", use_container_width=True):
                    st.session_state["ticket_actions"][selected_ticket_id] = "ESCALATED_TIER_2"
                    st.warning(f"Ticket escalated to Senior Tier-2 Specialist!")
                    st.rerun()

            with b_col4:
                if st.button("🔒 Close Ticket", key=f"close_{selected_ticket_id}", use_container_width=True):
                    st.session_state["ticket_actions"][selected_ticket_id] = "CLOSED"
                    st.info(f"Ticket closed.")
                    st.rerun()

            # Internal Notes Box
            st.markdown("---")
            st.markdown("##### 📝 Internal Support Notes & Audit Trail:")
            existing_note = st.session_state["ticket_notes"].get(selected_ticket_id, "")
            new_note = st.text_input(
                "Add internal note:",
                value=existing_note,
                placeholder="e.g. Verified tracking with courier; customer requested callback...",
                key=f"note_input_{selected_ticket_id}"
            )
            if new_note != existing_note:
                st.session_state["ticket_notes"][selected_ticket_id] = new_note
