# graph_pipeline.py
"""
NBA from Call - LangGraph Agent 版 Pipeline
- 使用 DeepSeek API
- 圖狀流程：抽取 → 建檔 → 載入回饋 → 生成 → Critic 審查 →（必要時）修正 → 定稿
- 含 JSON 解析修復與重試
- revise 只輸出改動的 leads，避免 max_tokens 截斷
"""
import json
import operator
import os
import re
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import OpenAI

from prompts import (
    EXTRACT_CALL_PROMPT,
    BUILD_PROFILE_PROMPT,
    GENERATE_LEADS_PROMPT,
)
import feedback_store

# ============================================================
# 初始化
# ============================================================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"
MAX_TOKENS = 8192


# ============================================================
# State 定義
# ============================================================
class NBAState(TypedDict):
    customer: dict
    extractions: Annotated[list, operator.add]
    profile: dict
    leads: dict
    feedback_context: dict
    review_notes: list
    final_leads: dict
    iteration: int


# ============================================================
# JSON 修復工具
# ============================================================
def _try_fix_json(content: str) -> str | None:
    """嘗試修復常見的 JSON 語法問題（含截斷）"""
    if not content:
        return None

    fixed = content.strip()

    # 1. 移除 markdown code fence
    if fixed.startswith("```"):
        lines = fixed.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        fixed = "\n".join(lines).strip()

    # 2. 移除尾隨逗號
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    # 3. 檢查是否有未閉合的字串
    in_string = False
    escaped = False
    for ch in fixed:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    # 4. 若字串未閉合（被截斷），補上閉合引號
    if in_string:
        fixed += '"'

    # 5. 移除尾端不完整的逗號
    fixed = fixed.rstrip().rstrip(",")

    # 6. 補上未閉合的括號
    open_braces = fixed.count("{") - fixed.count("}")
    open_brackets = fixed.count("[") - fixed.count("]")

    if open_brackets > 0 or open_braces > 0:
        fixed += "]" * max(0, open_brackets)
        fixed += "}" * max(0, open_braces)

    return fixed


# ============================================================
# LLM 呼叫（含重試與修復）
# ============================================================
def call_llm(
    prompt: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    last_content = ""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            last_content = content

            # Debug: token 用量
            if hasattr(resp, "usage") and resp.usage:
                print(
                    f"      [tokens] prompt={resp.usage.prompt_tokens}, "
                    f"completion={resp.usage.completion_tokens}, "
                    f"total={resp.usage.total_tokens}"
                )

            if not content.strip():
                print(f"    ⚠️ 空回應（{attempt + 1}/{max_retries}），重試...")
                time.sleep(1)
                continue

            finish_reason = resp.choices[0].finish_reason
            if finish_reason == "length":
                print(f"    ⚠️ 回應被截斷（{len(content)} 字元），嘗試修復...")
                fixed = _try_fix_json(content)
                if fixed:
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
                time.sleep(1)
                continue

            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"    ⚠️ JSON 解析失敗（{attempt + 1}/{max_retries}）：{e}")
            print(f"       結尾：...{last_content[-150:]}")
            fixed = _try_fix_json(last_content)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
            time.sleep(1)

    raise ValueError(
        f"連續 {max_retries} 次無法解析 JSON，最後回應結尾：...{last_content[-200:]}"
    )


# ============================================================
# Node 1: 抽取所有通話
# ============================================================
def extract_all_calls(state: NBAState) -> dict:
    customer = state["customer"]
    extractions = []
    for call in customer["calls"]:
        print(f"    抽取 {call['call_id']}...")
        prompt = EXTRACT_CALL_PROMPT.replace("{transcript}", call["transcript"])
        result = call_llm(prompt)
        result["call_id"] = call["call_id"]
        result["date"] = call["date"]
        extractions.append(result)
    return {"extractions": extractions}


# ============================================================
# Node 2: 建立客戶檔案
# ============================================================
def build_customer_profile(state: NBAState) -> dict:
    print(f"    建立客戶檔案...")
    prompt = BUILD_PROFILE_PROMPT.replace(
        "{extractions}",
        json.dumps(state["extractions"], ensure_ascii=False, indent=2),
    )
    profile = call_llm(prompt)
    profile["customer_id"] = state["customer"]["customer_id"]
    profile["name"] = state["customer"]["name"]
    return {"profile": profile}


# ============================================================
# Node 3: 載入 RM 回饋作為 context
# ============================================================
def load_feedback_context(state: NBAState) -> dict:
    adopted = feedback_store.get_adopted_examples(limit=5)
    rejected = feedback_store.get_rejected_examples(limit=5)
    if adopted or rejected:
        print(f"    載入回饋：{len(adopted)} 採納、{len(rejected)} 拒絕")
    return {
        "feedback_context": {
            "adopted_examples": adopted,
            "rejected_examples": rejected,
        }
    }


# ============================================================
# Node 4: 生成 leads（考慮回饋）
# ============================================================
def generate_leads_node(state: NBAState) -> dict:
    print(f"    生成 leads...")
    feedback = state.get("feedback_context", {})
    feedback_hint = ""

    if feedback.get("adopted_examples"):
        feedback_hint += "\n【RM 過去採納的 leads（可作為風格參考）】\n"
        feedback_hint += json.dumps(
            feedback["adopted_examples"], ensure_ascii=False, indent=2
        )
    if feedback.get("rejected_examples"):
        feedback_hint += "\n【RM 過去拒絕的 leads（避免類似類型或補強證據）】\n"
        feedback_hint += json.dumps(
            feedback["rejected_examples"], ensure_ascii=False, indent=2
        )

    prompt = GENERATE_LEADS_PROMPT.replace(
        "{profile}",
        json.dumps(state["profile"], ensure_ascii=False, indent=2),
    ).replace(
        "{extractions}",
        json.dumps(state["extractions"], ensure_ascii=False, indent=2),
    ) + feedback_hint

    leads = call_llm(prompt)
    return {"leads": leads, "iteration": state.get("iteration", 0) + 1}


# ============================================================
# Node 5: 自我審查（Critic Agent）- 放寬標準
# ============================================================
CRITIC_PROMPT = """你是財富管理品質審查員。
請審查以下 leads。

【判斷標準 - 只在「嚴重問題」時才要求修正】
嚴重問題（needs_revision = true）：
- supporting_evidence 引用了不存在的 call_id
- rationale 完全無通話依據（明顯幻覺）
- suggested_action 完全不可執行或不明確

可接受（needs_revision = false）：
- 用詞可以更好
- 可以補充更多細節
- 信心度可以微調
- 順序可以調整
- 產品建議可以優化

【原則】
寧可放過小問題，也不要過度要求修正。只有在確實有嚴重問題時才回傳 needs_revision = true。

輸出 JSON：
{
  "issues": [{"lead_id": "", "issue": "", "severity": "high|medium|low"}],
  "overall_quality": 0.0,
  "needs_revision": false,
  "revision_notes": ""
}

Leads：
{leads}

原始通話證據：
{extractions}
"""


def critic_node(state: NBAState) -> dict:
    print(f"    Critic 審查...")
    prompt = CRITIC_PROMPT.replace(
        "{leads}",
        json.dumps(state["leads"], ensure_ascii=False, indent=2),
    ).replace(
        "{extractions}",
        json.dumps(state["extractions"], ensure_ascii=False, indent=2),
    )
    review = call_llm(prompt, temperature=0.1)
    quality = review.get("overall_quality", "N/A")
    needs_rev = review.get("needs_revision", False)
    issues = review.get("issues", [])
    print(f"      品質分數：{quality}，需修正：{needs_rev}（{len(issues)} 個問題）")
    return {"review_notes": [review]}


# ============================================================
# Node 6: 修正 leads（只輸出改動部分，避免截斷）
# ============================================================
REVISE_PROMPT = """你是財富管理顧問。
根據品質審查員的意見，修正以下 leads。

【審查意見】
{review}

【原始 leads】
{leads}

【原始通話證據】
{extractions}

【輸出要求 - 非常重要】
1. 只輸出「有修改」的 leads，沒改的不要重複輸出
2. 若某個 lead 審查員認為沒問題，不要包含在輸出中
3. 若所有 lead 都不需要修正，回傳空陣列
4. 每個輸出的 lead 必須包含完整欄位（lead_id, category, rationale, suggested_action, priority, confidence, supporting_evidence, suggested_products, talking_points）

輸出 JSON：
{{
  "revised_leads": [
    {{
      "lead_id": "L1",
      "category": "...",
      "rationale": "...",
      "suggested_action": "...",
      "priority": "高|中|低",
      "confidence": 0.0,
      "supporting_evidence": ["C001_2"],
      "suggested_products": [],
      "talking_points": []
    }}
  ]
}}
"""


def revise_node(state: NBAState) -> dict:
    print(f"    修正 leads...")
    review = state["review_notes"][-1]
    prompt = REVISE_PROMPT.replace(
        "{review}",
        json.dumps(review, ensure_ascii=False, indent=2),
    ).replace(
        "{leads}",
        json.dumps(state["leads"], ensure_ascii=False, indent=2),
    ).replace(
        "{extractions}",
        json.dumps(state["extractions"], ensure_ascii=False, indent=2),
    )
    result = call_llm(prompt, temperature=0.3, max_tokens=8192)

    # 合併：修正版覆蓋原版，沒改的保留
    revised_leads = result.get("revised_leads", [])
    original_container = state["leads"]
    original = (
        original_container.get("leads", [])
        if isinstance(original_container, dict)
        else []
    )

    if not revised_leads:
        merged = original_container
        print(f"      （無改動）")
    else:
        revised_map = {l.get("lead_id"): l for l in revised_leads}
        merged_leads = [revised_map.get(l.get("lead_id"), l) for l in original]
        merged = {**original_container, "leads": merged_leads}
        print(f"      （修正了 {len(revised_leads)}/{len(original)} 個 leads）")

    return {
        "final_leads": merged,
        "leads": merged,
        "iteration": state.get("iteration", 0) + 1,
    }


# ============================================================
# 路由決策
# ============================================================
def route_after_critic(state: NBAState) -> str:
    review = state["review_notes"][-1]
    needs_rev = review.get("needs_revision", False)
    iteration = state.get("iteration", 0)

    # 最多迭代 3 次（含初次生成），避免無限迴圈
    if needs_rev and iteration < 3:
        return "revise"
    if needs_rev:
        print(f"      ⚠️ 已達最大迭代次數（{iteration}），強制定稿")
    return "finalize"


# ============================================================
# Node 7: 定稿
# ============================================================
def finalize_node(state: NBAState) -> dict:
    if state.get("final_leads"):
        return {}
    return {"final_leads": state["leads"]}


# ============================================================
# 建立 Graph
# ============================================================
def build_graph():
    graph = StateGraph(NBAState)

    graph.add_node("extract", extract_all_calls)
    graph.add_node("profile", build_customer_profile)
    graph.add_node("load_feedback", load_feedback_context)
    graph.add_node("generate", generate_leads_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "profile")
    graph.add_edge("profile", "load_feedback")
    graph.add_edge("load_feedback", "generate")
    graph.add_edge("generate", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"revise": "revise", "finalize": "finalize"},
    )
    graph.add_edge("revise", "critic")
    graph.add_edge("finalize", END)

    return graph.compile()


# ============================================================
# 主流程
# ============================================================
def run_graph_pipeline(data_path: str = "data/synthetic_calls.json") -> dict:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    app = build_graph()
    results = {}
    total = len(data["customers"])

    for idx, customer in enumerate(data["customers"], 1):
        cid = customer["customer_id"]
        print(f"\n=== [{idx}/{total}] [LangGraph] 處理 {cid} {customer['name']} ===")

        initial_state: NBAState = {
            "customer": customer,
            "extractions": [],
            "profile": {},
            "leads": {},
            "feedback_context": {},
            "review_notes": [],
            "final_leads": {},
            "iteration": 0,
        }

        try:
            final_state = app.invoke(initial_state)

            review = (
                final_state["review_notes"][-1]
                if final_state["review_notes"]
                else {}
            )
            final_leads_container = final_state.get("final_leads", {})
            n_leads = (
                len(final_leads_container.get("leads", []))
                if isinstance(final_leads_container, dict)
                else 0
            )

            print(f"  ✅ 完成，產出 {n_leads} 個 leads")
            print(f"     品質分數：{review.get('overall_quality', 'N/A')}")
            print(f"     修正次數：{max(0, final_state.get('iteration', 0) - 1)}")

            results[cid] = {
                "customer": customer,
                "extractions": final_state["extractions"],
                "profile": final_state["profile"],
                "leads": final_state["final_leads"],
                "critic_reviews": final_state["review_notes"],
            }

        except Exception as e:
            # 單一客戶失敗不中斷整個流程
            print(f"  ❌ {cid} 處理失敗：{e}")
            results[cid] = {
                "customer": customer,
                "extractions": [],
                "profile": {},
                "leads": {"leads": []},
                "critic_reviews": [],
                "error": str(e),
            }

    with open("output_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 統計
    success = sum(1 for r in results.values() if "error" not in r)
    print(f"\n✅ LangGraph pipeline 完成")
    print(f"   成功：{success}/{total}")
    print(f"   結果寫入 output_results.json")
    return results


if __name__ == "__main__":
    run_graph_pipeline()