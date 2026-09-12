"""
View: RAG Knowledge Search
Semantic (all-MiniLM-L6-v2) and lexical (TF-IDF) knowledge retrieval engine over
2,400+ historical Amazon support resolutions, with policy SOP cheatsheets and on-the-fly grounded reply generation.
"""

import streamlit as st

POLICIES = [
    {
        "title": "💰 Refund Authorization & SLA Timelines",
        "content": """
        - **Credit Cards**: 3–5 business days after warehouse scan. Up to 1–2 billing cycles depending on financial institution.
        - **Debit Cards**: Up to 10 business days.
        - **Amazon Gift Card Balance**: 2–4 hours after return package carrier scan.
        - **Restocking Fees**: Waived for defective, damaged, or incorrect shipments.
        - **Specialist Authorization**: Required for transactions exceeding $100.00 or manual balance adjustments.
        """
    },
    {
        "title": "📦 Delayed & Lost in Transit Protocol",
        "content": """
        - **Carrier Delay Window**: Advise customer to allow 48 hours beyond the promised delivery date before replacement dispatch.
        - **Carrier Marking 'Delivered' (Missing Box)**: Check porch, side doors, neighbors, building mailroom. Wait 36 hours for premature carrier scan.
        - **Damaged Packages**: Request photo of damaged item and shipping label; offer instant replacement or prepaid return locker drop-off.
        """
    },
    {
        "title": "🔐 Account Access & 2-Factor Authentication Reset",
        "content": """
        - **Disconnected Phone / Lost Authenticator**: Never request passwords or full credit card numbers in public or DM.
        - **Account Recovery Protocol**: Direct user to `amzn.to/account-recovery` for secure Two-Step Verification Account Recovery with government-issued photo ID.
        - **Unauthorized Charges**: Flag customer account for security review; guide customer to Order History to verify unauthorized digital subscriptions.
        """
    },
    {
        "title": "🔄 Return Window & Drop-Off Locations",
        "content": """
        - **Standard Window**: 30 days from delivery receipt.
        - **Holiday Policy**: Items purchased between Oct 11 and Dec 31 can be returned until Jan 31 of the following year.
        - **Drop-Off Partners**: Whole Foods Market, Kohl's, Amazon Fresh, and Amazon Lockers (no box or label required).
        """
    }
]


def render_rag_search(pipeline, retrieval_mode: str = "hybrid"):
    st.markdown('<div class="main-header">📚 RAG Knowledge Base & Precedent Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Search over 2,400+ historical AmazonHelp resolution precedents using hybrid semantic vector search and exact keyword matching.</div>',
        unsafe_allow_html=True
    )

    # Search Bar & Controls
    search_col1, search_col2 = st.columns([3.5, 1.5])
    with search_col1:
        search_query = st.text_input(
            "Search Knowledge Precedents:",
            placeholder="e.g. where is my delayed package, return damaged shirt, accidental prime video charge...",
            key="rag_search_query"
        )

    with search_col2:
        search_btn = st.button("🔍 Search Precedents", type="primary", use_container_width=True)

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        mode_option = st.selectbox(
            "Search Retrieval Mode:",
            ["Hybrid (Dense MiniLM + TF-IDF RRF)", "Sparse (TF-IDF Lexical Only)", "Dense (Semantic Only)"],
            index=0 if retrieval_mode == "hybrid" else 1,
            key="rag_mode_select"
        )
        active_mode = "hybrid" if "Hybrid" in mode_option else ("dense" if "Dense" in mode_option else "sparse")

    with f_col2:
        intent_filter = st.selectbox(
            "Filter by Intent Domain:",
            ["All Intents", "order_delivery_delay", "refund_request", "billing_issue", "account_access", "cancellation_request", "technical_bug", "complaint_escalation", "general_inquiry"],
            index=0,
            key="rag_intent_filter"
        )

    with f_col3:
        top_k = st.slider("Top Results (k):", min_value=1, max_value=10, value=4, key="rag_top_k")

    # Perform Search
    if search_btn or search_query:
        query_text = search_query.strip()
        if not query_text:
            st.warning("Please type a search query above.")
        else:
            with st.spinner(f"Searching resolution archive using {active_mode.upper()} mode..."):
                retriever = pipeline.retriever
                # Fetch more if intent filtering is applied
                fetch_k = top_k * 3 if intent_filter != "All Intents" else top_k
                cases = retriever.retrieve(query_text, k=fetch_k, mode=active_mode)

                if intent_filter != "All Intents":
                    cases = [c for c in cases if c.get("intent") == intent_filter][:top_k]

            st.markdown("---")

            # Grounded Answer Generation Option
            col_ans1, col_ans2 = st.columns([2.5, 1.5])
            with col_ans1:
                st.markdown(f"### 🎯 Search Results ({len(cases)} matched precedents)")
            with col_ans2:
                generate_answer = st.button("✨ Generate Grounded Answer for Query", use_container_width=True)

            if generate_answer and cases:
                gen_result = pipeline.generator.generate(query_text, cases[0]["intent"], cases)
                st.markdown(
                    f"""
                    <div style="background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                        <div style="font-size: 0.8rem; font-weight: bold; color: #166534; text-transform: uppercase; margin-bottom: 0.3rem;">
                            ✨ Instant Grounded AI Resolution
                        </div>
                        <div style="font-size: 1rem; color: #14532D; line-height: 1.6;">
                            {gen_result["reply_text"]}
                        </div>
                        <div style="font-size: 0.75rem; color: #15803D; margin-top: 0.5rem;">
                            Grounded in Historical Cases: <code>{', '.join(gen_result['grounding_sources'])}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if not cases:
                st.info("No matching historical precedent found meeting criteria. Try loosening search terms or clearing intent filter.")
            else:
                for idx, case in enumerate(cases):
                    sim = case.get("similarity_score", 0.0)
                    dense = case.get("dense_score", 0.0)
                    sparse = case.get("sparse_score", 0.0)
                    intent = case.get("intent", "general_inquiry")

                    st.markdown(
                        f"""
                        <div class="evidence-card" style="margin-bottom: 1rem; padding: 1.1rem; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <div>
                                    <span style="font-weight: 700; font-size: 1rem; color: #1E293B;">Precedent #{idx+1}</span>
                                    <span class="tag" style="margin-left: 0.5rem;">ID: {case.get('conversation_id')}</span>
                                    <span class="tag" style="background: #FEF3C7; color: #92400E;">{intent}</span>
                                </div>
                                <div>
                                    <span style="background: #EEF2FF; color: #4338CA; padding: 3px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;">
                                        Match Score: {sim*100:.1f}%
                                    </span>
                                </div>
                            </div>
                            <div style="font-size: 0.75rem; color: #64748B; margin-bottom: 0.6rem;">
                                Score Breakdown: Dense Semantic: <code>{dense*100:.0f}%</code> | Sparse TF-IDF: <code>{sparse*100:.0f}%</code> | Engine: <code>{case.get('retrieval_mode', active_mode).upper()}</code>
                            </div>
                            <div style="margin-bottom: 0.6rem;">
                                <strong style="color: #475569;">Inbound Customer Tweet:</strong><br/>
                                <div style="color: #1E293B; background: #F8FAFC; padding: 0.6rem; border-radius: 6px; margin-top: 0.2rem; border-left: 3px solid #94A3B8;">
                                    "{case.get('customer_text')}"
                                </div>
                            </div>
                            <div>
                                <strong style="color: #059669;">Verified AmazonHelp Resolution:</strong><br/>
                                <div style="color: #064E3B; background: #F0FDF4; padding: 0.6rem; border-radius: 6px; margin-top: 0.2rem; border-left: 3px solid #10B981;">
                                    "{case.get('brand_reply_text')}"
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # Section 2: Policy Standard Operating Procedures (SOPs)
    st.markdown("---")
    st.markdown("### 📖 Official AmazonHelp Policy SOP Cheatsheets")
    st.caption("Standard operating procedure guidelines for customer dispute resolution and SLA compliance.")

    pol_col1, pol_col2 = st.columns(2)
    for i, pol in enumerate(POLICIES):
        target_col = pol_col1 if i % 2 == 0 else pol_col2
        with target_col:
            with st.expander(pol["title"], expanded=(i == 0)):
                st.markdown(pol["content"])
