# app.py
"""
Next Best Action from Call - RM Dashboard
Full version: includes RM feedback loop
"""
import json
import os
from datetime import datetime

import streamlit as st

import feedback_store

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="NBA from Call - RM Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Load data
# ============================================================
RESULTS_PATH = "output_results.json"
EVAL_PATH = "evaluation_report.json"


@st.cache_data(ttl=10)
def load_results(path: str = RESULTS_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=10)
def load_eval_report(path: str = EVAL_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


results = load_results()

if not results:
    st.error(
        "`output_results.json` not found.\n\n"
        "Please run the following first:\n"
        "- `python graph_pipeline.py` (LangGraph agent version)"
    )
    st.stop()


# ============================================================
# Session state init
# ============================================================
if "rm_name" not in st.session_state:
    st.session_state["rm_name"] = "RM_001"

if "rejecting" not in st.session_state:
    st.session_state["rejecting"] = {}


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("NBA from Call")
st.sidebar.caption("Next Best Action from Call Records")

# --- RM identity ---
st.sidebar.markdown("### RM Identity")
st.session_state["rm_name"] = st.sidebar.text_input(
    "RM Name",
    value=st.session_state["rm_name"],
    help="Used to tag feedback source",
)

st.sidebar.markdown("---")

# --- Customer selector ---
customer_options = {
    cid: f"{r['customer']['name']} ({cid})"
    for cid, r in results.items()
}
selected = st.sidebar.selectbox(
    "Select Customer",
    options=list(customer_options.keys()),
    format_func=lambda x: customer_options[x],
)

st.sidebar.markdown("---")

# --- Evaluation metrics ---
st.sidebar.markdown("### System Evaluation")
eval_report = load_eval_report()
if eval_report:
    stats = eval_report.get("base_evaluation", {}).get("stats", {})
    if stats:
        st.sidebar.metric("Total Leads", stats.get("total_leads", 0))
        st.sidebar.metric(
            "Evidence Valid Rate",
            f"{stats.get('evidence_valid_rate', 0) * 100:.0f}%",
        )
        st.sidebar.metric(
            "Grounded Rate",
            f"{stats.get('grounded_rate', 0) * 100:.0f}%",
        )
        st.sidebar.metric(
            "Hallucination Risk",
            f"{stats.get('hallucination_rate', 0) * 100:.0f}%",
        )
        st.sidebar.metric(
            "English Compliance",
            f"{stats.get('language_compliance_rate', 1.0) * 100:.0f}%",
        )
        st.sidebar.metric(
            "Priority Consistency",
            f"{stats.get('priority_consistency_rate', 1.0) * 100:.0f}%",
        )
else:
    st.sidebar.info("No evaluation report yet.\nRun `python evaluate.py`.")

st.sidebar.markdown("---")

# --- Feedback summary ---
fb_summary = feedback_store.get_feedback_summary()
st.sidebar.markdown("### RM Feedback")
st.sidebar.metric("Total Feedback", fb_summary["total"])
if fb_summary["total"] > 0:
    st.sidebar.metric("Adoption Rate", f"{fb_summary['adopt_rate'] * 100:.0f}%")
    col_a, col_b = st.sidebar.columns(2)
    col_a.metric("Adopted", fb_summary["adopt"])
    col_b.metric("Rejected", fb_summary["reject"])
else:
    st.sidebar.caption("No feedback yet")

st.sidebar.markdown("---")
st.sidebar.caption("POC Demo | Synthetic Data | Internal Only")


# ============================================================
# Main content
# ============================================================
r = results[selected]
customer = r["customer"]
profile = r.get("profile", {})
leads_data = r.get("leads", {})
leads = leads_data.get("leads", []) if isinstance(leads_data, dict) else []
extractions = r.get("extractions", [])
critic_reviews = r.get("critic_reviews", [])

# --- Header ---
st.title(f"{customer['name']}")
st.caption(
    f"Customer ID: {customer['customer_id']} | "
    f"Age: {customer['age']} | "
    f"Occupation: {customer['occupation']} | "
    f"Calls: {len(customer['calls'])}"
)

# --- Two-column layout ---
col1, col2 = st.columns([1, 2], gap="large")


# ============================================================
# Left column: Customer profile
# ============================================================
with col1:
    st.subheader("Customer Profile")

    p = profile.get("profile", {})
    st.markdown(f"**Life Stage:** {p.get('life_stage', '-')}")
    st.markdown(f"**Risk Preference:** {p.get('risk_preference', '-')}")
    st.markdown(f"**Liquidity Need:** {p.get('liquidity_need', '-')}")
    st.markdown(f"**Estimated Capacity:** {p.get('estimated_capacity', '-')}")

    locations = p.get("locations_of_interest", [])
    if locations:
        st.markdown(f"**Locations of Interest:** {', '.join(locations)}")

    st.markdown("---")

    # --- Needs timeline ---
    st.markdown("### Needs Timeline")
    needs_timeline = profile.get("needs_timeline", [])
    if needs_timeline:
        for n in needs_timeline:
            urgency = n.get("urgency", "medium")
            urgency_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(urgency, "⚪")
            st.markdown(
                f"{urgency_icon} **{n.get('need', '-')}**  \n"
                f"  Urgency: {urgency} | Horizon: {n.get('horizon', '-')}  \n"
                f"  Evidence: `{', '.join(n.get('evidence', []))}`"
            )
    else:
        st.caption("No clear needs timeline")

    # --- Contradictions ---
    contradictions = profile.get("contradictions", [])
    if contradictions:
        st.markdown("---")
        st.markdown("### Contradictions")
        for c in contradictions:
            st.warning(
                f"**{c.get('signal_a', '')}** vs **{c.get('signal_b', '')}**\n\n"
                f"{c.get('note', '')}"
            )

    # --- Critic review (if available) ---
    if critic_reviews:
        st.markdown("---")
        st.markdown("### Critic Review")
        latest_review = critic_reviews[-1]
        quality = latest_review.get("overall_quality", None)
        if quality is not None:
            st.metric("Quality Score", f"{quality:.2f}")
        issues = latest_review.get("issues", [])
        if issues:
            with st.expander(f"Found {len(issues)} issue(s)", expanded=False):
                for issue in issues:
                    severity = issue.get("severity", "low")
                    icon = {
                        "high": "🔴", "medium": "🟡", "low": "🟢"
                    }.get(severity, "⚪")
                    st.markdown(
                        f"{icon} **[{issue.get('lead_id', '-')}]** "
                        f"{issue.get('issue', '')}"
                    )
        else:
            st.success("No issues found")
        st.caption(f"Revision iterations: {len(critic_reviews) - 1}")


# ============================================================
# Right column: Next Best Actions
# ============================================================
with col2:
    st.subheader("Next Best Actions")

    if not leads:
        st.info("No leads generated for this customer yet.")
    else:
        # Sort by priority: high > medium > low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        leads_sorted = sorted(
            leads,
            key=lambda x: priority_order.get(x.get("priority"), 3),
        )

        n_high = sum(1 for l in leads if l.get("priority") == "high")
        n_mid = sum(1 for l in leads if l.get("priority") == "medium")
        n_low = sum(1 for l in leads if l.get("priority") == "low")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Leads", len(leads))
        m2.metric("🔴 High", n_high)
        m3.metric("🟡 Medium", n_mid)
        m4.metric("🟢 Low", n_low)

        st.markdown("---")

        for lead in leads_sorted:
            lead_id = lead.get("lead_id", "?")
            category = lead.get("category", "Uncategorized")
            priority = lead.get("priority", "medium")
            confidence = lead.get("confidence", 0.0)

            priority_display = (
                priority if priority in ("high", "medium", "low") else "medium"
            )
            priority_icon = {
                "high": "🔴", "medium": "🟡", "low": "🟢"
            }.get(priority_display, "⚪")

            expander_title = (
                f"{priority_icon} [{priority_display}] {category} "
                f"— confidence {confidence:.0%}"
            )

            with st.expander(expander_title, expanded=(priority_display == "high")):
                st.markdown("**Rationale**")
                st.markdown(lead.get("rationale", "(none)"))

                st.markdown("**Suggested Action**")
                st.info(lead.get("suggested_action", "(none)"))

                talking_points = lead.get("talking_points", [])
                if talking_points:
                    st.markdown("**RM Talking Points**")
                    for tp in talking_points:
                        st.markdown(f"> {tp}")

                col_a, col_b = st.columns(2)
                with col_a:
                    evidence = lead.get("supporting_evidence", [])
                    st.markdown(f"**Evidence:** `{', '.join(evidence) or '(none)'}`")
                with col_b:
                    products = lead.get("suggested_products", [])
                    st.markdown(f"**Products:** {', '.join(products) or '(none)'}")

                st.progress(
                    min(max(confidence, 0.0), 1.0),
                    text=f"Confidence {confidence:.0%}",
                )

                # ============================================
                # RM feedback block
                # ============================================
                st.markdown("---")
                st.markdown("**RM Feedback**")

                existing_fb = feedback_store.get_feedback_for_lead(selected, lead_id)
                lead_key = f"{selected}_{lead_id}"

                if existing_fb:
                    if existing_fb["action"] == "adopt":
                        st.success(
                            f"Adopted | RM: {existing_fb['rm_name']} | "
                            f"{existing_fb['timestamp'][:16]}"
                        )
                    else:
                        st.warning(
                            f"Rejected | RM: {existing_fb['rm_name']} | "
                            f"{existing_fb['timestamp'][:16]}"
                        )
                        if existing_fb.get("reason"):
                            st.caption(f"Reason: {existing_fb['reason']}")

                    if st.button(
                        "Re-feedback",
                        key=f"redo_{lead_key}",
                        help="Undo current feedback and choose again",
                    ):
                        st.session_state[f"redo_confirm_{lead_key}"] = True

                    if st.session_state.get(f"redo_confirm_{lead_key}"):
                        st.warning("Confirm undoing this feedback?")
                        c1, c2 = st.columns(2)
                        if c1.button("Confirm Undo", key=f"confirm_redo_{lead_key}"):
                            feedback_store.remove_feedback(selected, lead_id)
                            st.session_state[f"redo_confirm_{lead_key}"] = False
                            st.rerun()
                        if c2.button("Cancel", key=f"cancel_redo_{lead_key}"):
                            st.session_state[f"redo_confirm_{lead_key}"] = False
                            st.rerun()

                else:
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        if st.button(
                            "Adopt",
                            key=f"adopt_{lead_key}",
                            use_container_width=True,
                            type="primary",
                        ):
                            feedback_store.record_feedback(
                                customer_id=selected,
                                lead_id=lead_id,
                                action="adopt",
                                reason="",
                                rm_name=st.session_state["rm_name"],
                            )
                            st.toast("Adopted", icon="✅")
                            st.rerun()

                    with btn_col2:
                        if st.button(
                            "Reject",
                            key=f"reject_{lead_key}",
                            use_container_width=True,
                        ):
                            st.session_state["rejecting"][lead_key] = True

                    if st.session_state["rejecting"].get(lead_key):
                        reason = st.text_input(
                            "Reason for rejection",
                            key=f"reason_input_{lead_key}",
                            placeholder="e.g., Customer handled it / Insufficient evidence / Product not suitable",
                        )
                        sub_c1, sub_c2 = st.columns(2)
                        with sub_c1:
                            if st.button(
                                "Submit Rejection",
                                key=f"submit_reject_{lead_key}",
                                use_container_width=True,
                            ):
                                feedback_store.record_feedback(
                                    customer_id=selected,
                                    lead_id=lead_id,
                                    action="reject",
                                    reason=reason,
                                    rm_name=st.session_state["rm_name"],
                                )
                                st.session_state["rejecting"][lead_key] = False
                                st.toast("Rejected", icon="❌")
                                st.rerun()
                        with sub_c2:
                            if st.button(
                                "Cancel",
                                key=f"cancel_reject_{lead_key}",
                                use_container_width=True,
                            ):
                                st.session_state["rejecting"][lead_key] = False
                                st.rerun()


# ============================================================
# Bottom: Call records and extraction results
# ============================================================
st.markdown("---")
st.subheader("Call Records & Extraction Results")

call_tabs = st.tabs(
    [f"{c['call_id']} | {c['date']}" for c in customer["calls"]]
)

for i, call in enumerate(customer["calls"]):
    with call_tabs[i]:
        tcol1, tcol2 = st.columns([3, 2])

        with tcol1:
            st.markdown("**Transcript**")
            st.text_area(
                "transcript",
                value=call["transcript"],
                height=300,
                label_visibility="collapsed",
                key=f"transcript_{selected}_{call['call_id']}",
            )

        with tcol2:
            st.markdown("**LLM Extraction**")
            ext = next(
                (e for e in extractions if e.get("call_id") == call["call_id"]),
                None,
            )
            if ext:
                st.json(ext)
            else:
                st.caption("No extraction yet")


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    f"NBA POC | RM: {st.session_state['rm_name']} | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)