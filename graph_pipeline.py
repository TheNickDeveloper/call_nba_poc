# graph_pipeline.py
"""
NBA from Call - LangGraph Agent Pipeline
- Uses DeepSeek API
- Graph flow: extract -> profile -> load_feedback -> generate -> critic -> (revise if needed) -> finalize
- Includes JSON repair and retry
- revise only outputs modified leads to avoid max_tokens truncation
- All prompts centralized in prompts.py
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
    CRITIC_PROMPT,
    REVISE_PROMPT,
)
import feedback_store
from confidence import apply_computed_confidence, latest_call_id

# ============================================================
# Init
# ============================================================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"
MAX_TOKENS = 8192


# ============================================================
# State Definition
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
# JSON repair utility
# ============================================================
def _try_fix_json(content: str) -> str | None:
    """Try to fix common JSON syntax issues (including truncation)."""
    if not content:
        return None

    fixed = content.strip()

    # 1. Remove markdown code fence
    if fixed.startswith("```"):
        lines = fixed.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        fixed = "\n".join(lines).strip()

    # 2. Remove trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    # 3. Check for unclosed strings
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

    # 4. Close unclosed string (truncated)
    if in_string:
        fixed += '"'

    # 5. Remove incomplete trailing comma
    fixed = fixed.rstrip().rstrip(",")

    # 6. Close unmatched braces
    open_braces = fixed.count("{") - fixed.count("}")
    open_brackets = fixed.count("[") - fixed.count("]")

    if open_brackets > 0 or open_braces > 0:
        fixed += "]" * max(0, open_brackets)
        fixed += "}" * max(0, open_braces)

    return fixed


# ============================================================
# English-output guard (deterministic, not prompt-dependent)
# ============================================================
# Keys intentionally kept in the customer's original language.
_ORIGINAL_LANGUAGE_KEYS = {"quote_original", "talking_points"}

_HAN = re.compile(r"[\u4e00-\u9fff]")

CORRECTION_SUFFIX = (
    "\n\n\u3010CORRECTION\u3011Your previous answer contained Chinese characters. "
    "Re-output the ENTIRE JSON in English only. Keep all JSON keys unchanged."
)


def _find_non_english(obj, path: str = "") -> list:
    """Return the JSON paths whose string value contains CJK characters.

    Keys in _ORIGINAL_LANGUAGE_KEYS are skipped by design.
    """
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _ORIGINAL_LANGUAGE_KEYS:
                continue
            hits += _find_non_english(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += _find_non_english(v, f"{path}[{i}]")
    elif isinstance(obj, str) and _HAN.search(obj):
        hits.append(path)
    return hits



# ============================================================
# LLM call (with retry and repair)
# ============================================================
def call_llm(
    prompt: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    last_content = ""
    correction = ""

    def _is_english(parsed) -> bool:
        """False (and arms a correction prompt) if the payload leaked Chinese."""
        nonlocal correction
        leaks = _find_non_english(parsed)
        if leaks:
            print(f"    [WARN] Non-English output at {len(leaks)} path(s): {leaks[:5]}")
            correction = CORRECTION_SUFFIX
            return False
        return True

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt + correction}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            last_content = content

            # Debug: token usage
            if hasattr(resp, "usage") and resp.usage:
                print(
                    f"      [tokens] prompt={resp.usage.prompt_tokens}, "
                    f"completion={resp.usage.completion_tokens}, "
                    f"total={resp.usage.total_tokens}"
                )

            if not content.strip():
                print(f"    [WARN] Empty response ({attempt + 1}/{max_retries}), retrying...")
                time.sleep(1)
                continue

            finish_reason = resp.choices[0].finish_reason
            if finish_reason == "length":
                print(f"    [WARN] Response truncated ({len(content)} chars), attempting repair...")
                fixed = _try_fix_json(content)
                if fixed:
                    try:
                        parsed = json.loads(fixed)
                    except json.JSONDecodeError:
                        parsed = None
                    if parsed is not None and _is_english(parsed):
                        return parsed
                time.sleep(1)
                continue

            parsed = json.loads(content)
            if _is_english(parsed):
                return parsed
            time.sleep(1)
            continue

        except json.JSONDecodeError as e:
            print(f"    [WARN] JSON parse failed ({attempt + 1}/{max_retries}): {e}")
            print(f"       tail: ...{last_content[-150:]}")
            fixed = _try_fix_json(last_content)
            if fixed:
                try:
                    parsed = json.loads(fixed)
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None and _is_english(parsed):
                    return parsed
            time.sleep(1)

    raise ValueError(
        f"Failed after {max_retries} retries (JSON parse or English-language check). "
        f"Last response tail: ...{last_content[-200:]}"
    )


# ============================================================
# Node 1: Extract all calls
# ============================================================
def extract_all_calls(state: NBAState) -> dict:
    customer = state["customer"]
    extractions = []
    for call in customer["calls"]:
        print(f"    Extracting {call['call_id']}...")
        prompt = EXTRACT_CALL_PROMPT.replace("{transcript}", call["transcript"])
        result = call_llm(prompt)
        result["call_id"] = call["call_id"]
        result["date"] = call["date"]
        extractions.append(result)
    return {"extractions": extractions}


# ============================================================
# Node 2: Build customer profile
# ============================================================
def build_customer_profile(state: NBAState) -> dict:
    print(f"    Building customer profile...")
    prompt = BUILD_PROFILE_PROMPT.replace(
        "{extractions}",
        json.dumps(state["extractions"], ensure_ascii=False, indent=2),
    )
    profile = call_llm(prompt)
    profile["customer_id"] = state["customer"]["customer_id"]
    profile["name"] = state["customer"]["name"]
    return {"profile": profile}


# ============================================================
# Node 3: Load RM feedback as context
# ============================================================
def load_feedback_context(state: NBAState) -> dict:
    adopted = feedback_store.get_adopted_examples(limit=5)
    rejected = feedback_store.get_rejected_examples(limit=5)
    if adopted or rejected:
        print(f"    Loaded feedback: {len(adopted)} adopted, {len(rejected)} rejected")
    return {
        "feedback_context": {
            "adopted_examples": adopted,
            "rejected_examples": rejected,
        }
    }


# ============================================================
# Deterministic confidence
# ============================================================
def _apply(leads_container, state: NBAState) -> None:
    """Recompute confidence/priority from evidence on a leads container."""
    if isinstance(leads_container, dict):
        apply_computed_confidence(
            leads_container.get("leads"),
            state.get("extractions", []),
            latest_call_id(state.get("customer", {})),
        )


# ============================================================
# Node 4: Generate leads (with feedback awareness)
# ============================================================
def generate_leads_node(state: NBAState) -> dict:
    print(f"    Generating leads...")
    feedback = state.get("feedback_context", {})
    feedback_hint = ""

    if feedback.get("adopted_examples"):
        feedback_hint += "\n[RM's previously ADOPTED leads (for style reference)]\n"
        feedback_hint += json.dumps(
            feedback["adopted_examples"], ensure_ascii=False, indent=2
        )
    if feedback.get("rejected_examples"):
        feedback_hint += "\n[RM's previously REJECTED leads (avoid similar patterns or strengthen evidence)]\n"
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
    _apply(leads, state)
    return {"leads": leads, "iteration": state.get("iteration", 0) + 1}


# ============================================================
# Node 5: Self-review (Critic Agent)
# ============================================================
def critic_node(state: NBAState) -> dict:
    print(f"    Critic review...")
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
    print(f"      Quality: {quality}, Needs revision: {needs_rev} ({len(issues)} issues)")
    return {"review_notes": [review]}


# ============================================================
# Node 6: Revise leads (only output modified ones)
# ============================================================
def revise_node(state: NBAState) -> dict:
    print(f"    Revising leads...")
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

    # Merge: revised versions overwrite originals, unchanged kept
    revised_leads = result.get("revised_leads", [])
    original_container = state["leads"]
    original = (
        original_container.get("leads", [])
        if isinstance(original_container, dict)
        else []
    )

    if not revised_leads:
        merged = original_container
        print(f"      (no changes)")
    else:
        revised_map = {l.get("lead_id"): l for l in revised_leads}
        merged_leads = [revised_map.get(l.get("lead_id"), l) for l in original]
        merged = {**original_container, "leads": merged_leads}
        print(f"      (revised {len(revised_leads)}/{len(original)} leads)")

    _apply(merged, state)

    return {
        "final_leads": merged,
        "leads": merged,
        "iteration": state.get("iteration", 0) + 1,
    }


# ============================================================
# Routing decision
# ============================================================
def route_after_critic(state: NBAState) -> str:
    review = state["review_notes"][-1]
    needs_rev = review.get("needs_revision", False)
    iteration = state.get("iteration", 0)

    # Max 3 iterations (including initial generation) to avoid infinite loop
    if needs_rev and iteration < 3:
        return "revise"
    if needs_rev:
        print(f"      [WARN] Max iterations reached ({iteration}), forcing finalize")
    return "finalize"


# ============================================================
# Node 7: Finalize
# ============================================================
def finalize_node(state: NBAState) -> dict:
    if state.get("final_leads"):
        return {}
    leads = state["leads"]
    _apply(leads, state)
    return {"final_leads": leads}


# ============================================================
# Build Graph
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
# Main pipeline
# ============================================================
def run_graph_pipeline(data_path: str = "data/synthetic_calls.json") -> dict:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    app = build_graph()
    results = {}
    total = len(data["customers"])

    for idx, customer in enumerate(data["customers"], 1):
        cid = customer["customer_id"]
        print(f"\n=== [{idx}/{total}] [LangGraph] Processing {cid} {customer['name']} ===")

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

            print(f"  [OK] Done, generated {n_leads} leads")
            print(f"       Quality: {review.get('overall_quality', 'N/A')}")
            print(f"       Revisions: {max(0, final_state.get('iteration', 0) - 1)}")

            results[cid] = {
                "customer": customer,
                "extractions": final_state["extractions"],
                "profile": final_state["profile"],
                "leads": final_state["final_leads"],
                "critic_reviews": final_state["review_notes"],
            }

        except Exception as e:
            # Single customer failure should not break the whole run
            print(f"  [FAIL] {cid} failed: {e}")
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

    # Summary
    success = sum(1 for r in results.values() if "error" not in r)
    print(f"\n[OK] LangGraph pipeline complete")
    print(f"     Success: {success}/{total}")
    print(f"     Results written to output_results.json")
    return results


if __name__ == "__main__":
    run_graph_pipeline()