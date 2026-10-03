# app.py
"""
Next Best Action from Call - RM Dashboard
完整版：含 RM 回饋閉環
"""
import json
import os
from datetime import datetime

import streamlit as st

import feedback_store

# ============================================================
# 頁面設定
# ============================================================
st.set_page_config(
    page_title="NBA from Call - RM Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 載入資料
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
        "⚠️ 找不到 `output_results.json`。\n\n"
        "請先執行以下其中一個指令產生結果：\n"
        "- `python pipeline.py`（簡單版）\n"
        "- `python graph_pipeline.py`（LangGraph agent 版）"
    )
    st.stop()


# ============================================================
# Session State 初始化
# ============================================================
if "rm_name" not in st.session_state:
    st.session_state["rm_name"] = "RM_001"

# 用於「不採納」原因輸入的展開狀態
if "rejecting" not in st.session_state:
    st.session_state["rejecting"] = {}  # {lead_key: True/False}


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("📞 NBA from Call")
st.sidebar.caption("Next Best Action from Call Records")

# --- RM 身份 ---
st.sidebar.markdown("### 👤 RM 身份")
st.session_state["rm_name"] = st.sidebar.text_input(
    "RM 名稱",
    value=st.session_state["rm_name"],
    help="用於記錄回饋來源",
)

st.sidebar.markdown("---")

# --- 客戶選擇 ---
customer_options = {
    cid: f"{r['customer']['name']} ({cid})"
    for cid, r in results.items()
}
selected = st.sidebar.selectbox(
    "選擇客戶",
    options=list(customer_options.keys()),
    format_func=lambda x: customer_options[x],
)

st.sidebar.markdown("---")

# --- 評估指標 ---
st.sidebar.markdown("### 📊 系統評估")
eval_report = load_eval_report()
if eval_report:
    stats = eval_report.get("base_evaluation", {}).get("stats", {})
    if stats:
        st.sidebar.metric("Leads 總數", stats.get("total_leads", 0))
        st.sidebar.metric(
            "Evidence 有效率",
            f"{stats.get('evidence_valid_rate', 0) * 100:.0f}%",
        )
        st.sidebar.metric(
            "Grounded 率",
            f"{stats.get('grounded_rate', 0) * 100:.0f}%",
        )
        st.sidebar.metric(
            "幻覺風險率",
            f"{stats.get('hallucination_rate', 0) * 100:.0f}%",
        )
else:
    st.sidebar.info("尚未產生評估報告\n請執行 `python evaluate.py`")

st.sidebar.markdown("---")

# --- 回饋統計 ---
fb_summary = feedback_store.get_feedback_summary()
st.sidebar.markdown("### 🗳️ RM 回饋統計")
st.sidebar.metric("總回饋數", fb_summary["total"])
if fb_summary["total"] > 0:
    st.sidebar.metric("採納率", f"{fb_summary['adopt_rate'] * 100:.0f}%")
    col_a, col_b = st.sidebar.columns(2)
    col_a.metric("✅ 採納", fb_summary["adopt"])
    col_b.metric("❌ 拒絕", fb_summary["reject"])
else:
    st.sidebar.caption("尚無回饋紀錄")

st.sidebar.markdown("---")
st.sidebar.caption(
    "POC Demo ｜ 合成資料 ｜ 僅供內部展示"
)


# ============================================================
# 主畫面
# ============================================================
r = results[selected]
customer = r["customer"]
profile = r.get("profile", {})
leads_data = r.get("leads", {})
leads = leads_data.get("leads", []) if isinstance(leads_data, dict) else []
extractions = r.get("extractions", [])
critic_reviews = r.get("critic_reviews", [])

# --- 標題區 ---
st.title(f"👤 {customer['name']}")
st.caption(
    f"客戶 ID：{customer['customer_id']} ｜ "
    f"年齡：{customer['age']} ｜ "
    f"職業：{customer['occupation']} ｜ "
    f"通話數：{len(customer['calls'])}"
)

# --- 兩欄版面 ---
col1, col2 = st.columns([1, 2], gap="large")


# ============================================================
# 左欄：客戶檔案
# ============================================================
with col1:
    st.subheader("📋 客戶檔案")

    p = profile.get("profile", {})
    st.markdown(f"**人生階段**：{p.get('life_stage', '-')}")
    st.markdown(f"**風險偏好**：{p.get('risk_preference', '-')}")
    st.markdown(f"**流動性需求**：{p.get('liquidity_need', '-')}")
    st.markdown(f"**資金量能**：{p.get('estimated_capacity', '-')}")

    locations = p.get("locations_of_interest", [])
    if locations:
        st.markdown(f"**關注地區**：{', '.join(locations)}")

    st.markdown("---")

    # --- 需求時間軸 ---
    st.markdown("### 🎯 需求時間軸")
    needs_timeline = profile.get("needs_timeline", [])
    if needs_timeline:
        for n in needs_timeline:
            urgency = n.get("urgency", "中")
            urgency_icon = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(urgency, "⚪")
            st.markdown(
                f"{urgency_icon} **{n.get('need', '-')}**  \n"
                f"　緊迫度：{urgency} ｜ 時間軸：{n.get('horizon', '-')}  \n"
                f"　證據：`{', '.join(n.get('evidence', []))}`"
            )
    else:
        st.caption("無明確需求時間軸")

    # --- 矛盾訊號 ---
    contradictions = profile.get("contradictions", [])
    if contradictions:
        st.markdown("---")
        st.markdown("### ⚠️ 矛盾訊號")
        for c in contradictions:
            st.warning(
                f"**{c.get('signal_a', '')}** vs **{c.get('signal_b', '')}**\n\n"
                f"{c.get('note', '')}"
            )

    # --- Critic 審查結果（若有） ---
    if critic_reviews:
        st.markdown("---")
        st.markdown("### 🔍 Critic 審查")
        latest_review = critic_reviews[-1]
        quality = latest_review.get("overall_quality", None)
        if quality is not None:
            st.metric("品質分數", f"{quality:.2f}")
        issues = latest_review.get("issues", [])
        if issues:
            with st.expander(f"發現 {len(issues)} 個問題", expanded=False):
                for issue in issues:
                    severity = issue.get("severity", "low")
                    icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
                    st.markdown(
                        f"{icon} **[{issue.get('lead_id', '-')}]** "
                        f"{issue.get('issue', '')}"
                    )
        else:
            st.success("✅ 未發現問題")
        st.caption(f"修正迭代次數：{len(critic_reviews) - 1}")


# ============================================================
# 右欄：Next Best Actions
# ============================================================
with col2:
    st.subheader("🚀 Next Best Actions")

    if not leads:
        st.info("此客戶尚無生成的 leads")
    else:
        # 排序：高 > 中 > 低
        priority_order = {"高": 0, "中": 1, "低": 2}
        leads_sorted = sorted(
            leads,
            key=lambda x: priority_order.get(x.get("priority"), 3),
        )

        # 統計列
        n_high = sum(1 for l in leads if l.get("priority") == "高")
        n_mid = sum(1 for l in leads if l.get("priority") == "中")
        n_low = sum(1 for l in leads if l.get("priority") == "低")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("總 Leads", len(leads))
        m2.metric("🔴 高優先", n_high)
        m3.metric("🟡 中優先", n_mid)
        m4.metric("🟢 低優先", n_low)

        st.markdown("---")

        # --- 逐個 Lead 卡片 ---
        for lead in leads_sorted:
            lead_id = lead.get("lead_id", "?")
            category = lead.get("category", "未分類")
            priority = lead.get("priority", "中")
            confidence = lead.get("confidence", 0.0)
            priority_icon = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(priority, "⚪")

            expander_title = (
                f"{priority_icon} [{priority}] {category} "
                f"— 信心度 {confidence:.0%}"
            )

            with st.expander(expander_title, expanded=(priority == "高")):
                # --- 為什麼推這個 ---
                st.markdown("**📌 為什麼推這個（Rationale）**")
                st.markdown(lead.get("rationale", "（無）"))

                # --- 建議行動 ---
                st.markdown("**✅ 建議行動（Suggested Action）**")
                st.info(lead.get("suggested_action", "（無）"))

                # --- 開場話術 ---
                talking_points = lead.get("talking_points", [])
                if talking_points:
                    st.markdown("**💬 RM 開場話術**")
                    for tp in talking_points:
                        st.markdown(f"> {tp}")

                # --- 證據與產品 ---
                col_a, col_b = st.columns(2)
                with col_a:
                    evidence = lead.get("supporting_evidence", [])
                    st.markdown(f"**📎 證據來源**：`{', '.join(evidence) or '（無）'}`")
                with col_b:
                    products = lead.get("suggested_products", [])
                    st.markdown(f"**🛍️ 建議產品**：{', '.join(products) or '（無）'}")

                # --- 信心度進度條 ---
                st.progress(
                    min(max(confidence, 0.0), 1.0),
                    text=f"信心度 {confidence:.0%}",
                )

                # ============================================
                # RM 回饋區塊
                # ============================================
                st.markdown("---")
                st.markdown("**🗳️ RM 回饋**")

                existing_fb = feedback_store.get_feedback_for_lead(selected, lead_id)
                lead_key = f"{selected}_{lead_id}"

                if existing_fb:
                    # 已回饋：顯示結果，並允許撤銷
                    if existing_fb["action"] == "adopt":
                        st.success(
                            f"✅ 已採納 ｜ RM: {existing_fb['rm_name']} ｜ "
                            f"{existing_fb['timestamp'][:16]}"
                        )
                    else:
                        st.warning(
                            f"❌ 已拒絕 ｜ RM: {existing_fb['rm_name']} ｜ "
                            f"{existing_fb['timestamp'][:16]}"
                        )
                        if existing_fb.get("reason"):
                            st.caption(f"原因：{existing_fb['reason']}")

                    if st.button(
                        "🔄 重新回饋",
                        key=f"redo_{lead_key}",
                        help="撤銷現有回饋並重新選擇",
                    ):
                        st.session_state[f"redo_confirm_{lead_key}"] = True

                    if st.session_state.get(f"redo_confirm_{lead_key}"):
                        st.warning("確定要撤銷這筆回饋嗎？")
                        c1, c2 = st.columns(2)
                        if c1.button("確認撤銷", key=f"confirm_redo_{lead_key}"):
                            feedback_store.remove_feedback(selected, lead_id)
                            st.session_state[f"redo_confirm_{lead_key}"] = False
                            st.rerun()
                        if c2.button("取消", key=f"cancel_redo_{lead_key}"):
                            st.session_state[f"redo_confirm_{lead_key}"] = False
                            st.rerun()

                else:
                    # 尚未回饋：顯示採納/拒絕按鈕
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        if st.button(
                            "✅ 採納",
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
                            st.toast("已記錄採納 ✅", icon="✅")
                            st.rerun()

                    with btn_col2:
                        if st.button(
                            "❌ 不採納",
                            key=f"reject_{lead_key}",
                            use_container_width=True,
                        ):
                            st.session_state["rejecting"][lead_key] = True

                    # --- 拒絕原因輸入 ---
                    if st.session_state["rejecting"].get(lead_key):
                        reason = st.text_input(
                            "不採納原因",
                            key=f"reason_input_{lead_key}",
                            placeholder="例如：客戶已自行處理 / 證據不足 / 產品不適合",
                        )
                        sub_c1, sub_c2 = st.columns(2)
                        with sub_c1:
                            if st.button(
                                "送出拒絕",
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
                                st.toast("已記錄不採納 ❌", icon="❌")
                                st.rerun()
                        with sub_c2:
                            if st.button(
                                "取消",
                                key=f"cancel_reject_{lead_key}",
                                use_container_width=True,
                            ):
                                st.session_state["rejecting"][lead_key] = False
                                st.rerun()


# ============================================================
# 底部：通話紀錄
# ============================================================
st.markdown("---")
st.subheader("📜 通話紀錄與抽取結果")

call_tabs = st.tabs(
    [f"{c['call_id']} ｜ {c['date']}" for c in customer["calls"]]
)

for i, call in enumerate(customer["calls"]):
    with call_tabs[i]:
        tcol1, tcol2 = st.columns([3, 2])

        with tcol1:
            st.markdown("**📝 逐字稿**")
            st.text_area(
                "transcript",
                value=call["transcript"],
                height=300,
                label_visibility="collapsed",
                key=f"transcript_{selected}_{call['call_id']}",
            )

        with tcol2:
            st.markdown("**🔍 LLM 抽取結果**")
            ext = next(
                (e for e in extractions if e.get("call_id") == call["call_id"]),
                None,
            )
            if ext:
                st.json(ext)
            else:
                st.caption("尚無抽取結果")


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    f"NBA POC ｜ RM: {st.session_state['rm_name']} ｜ "
    f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}"
)