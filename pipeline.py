# pipeline.py
"""
NBA from Call - 簡單版 Pipeline
- 使用 DeepSeek API
- 線性流程：抽取 → 建檔 → 生成 leads
- 含 JSON 解析修復與重試
"""
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from prompts import (
    EXTRACT_CALL_PROMPT,
    BUILD_PROFILE_PROMPT,
    GENERATE_LEADS_PROMPT,
)

# ============================================================
# 初始化
# ============================================================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"
MAX_TOKENS = 8000


# ============================================================
# JSON 修復工具
# ============================================================
def _try_fix_json(content: str) -> str | None:
    """嘗試修復常見的 JSON 語法問題"""
    if not content:
        return None

    fixed = content.strip()

    # 移除 markdown code fence（```json ... ```）
    if fixed.startswith("```"):
        lines = fixed.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        fixed = "\n".join(lines).strip()

    # 移除尾隨逗號（,} 或 ,]）
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    # 若被截斷，嘗試補上結尾
    if not fixed.endswith("}"):
        open_braces = fixed.count("{") - fixed.count("}")
        open_brackets = fixed.count("[") - fixed.count("]")
        fixed += "]" * max(0, open_brackets)
        fixed += "}" * max(0, open_braces)

    return fixed


# ============================================================
# LLM 呼叫（含重試與修復）
# ============================================================
def call_llm(prompt: str, temperature: float = 0.2, max_retries: int = 3) -> dict:
    last_content = ""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )
            content = resp.choices[0].message.content or ""
            last_content = content

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
# 步驟 1：抽取單通電話
# ============================================================
def extract_call(call: dict) -> dict:
    prompt = EXTRACT_CALL_PROMPT.replace("{transcript}", call["transcript"])
    result = call_llm(prompt)
    result["call_id"] = call["call_id"]
    result["date"] = call["date"]
    return result


# ============================================================
# 步驟 2：建立客戶檔案
# ============================================================
def build_profile(customer: dict, extractions: list[dict]) -> dict:
    prompt = BUILD_PROFILE_PROMPT.replace(
        "{extractions}",
        json.dumps(extractions, ensure_ascii=False, indent=2),
    )
    profile = call_llm(prompt)
    profile["customer_id"] = customer["customer_id"]
    profile["name"] = customer["name"]
    return profile


# ============================================================
# 步驟 3：生成 leads
# ============================================================
def generate_leads(profile: dict, extractions: list[dict]) -> dict:
    prompt = GENERATE_LEADS_PROMPT.replace(
        "{profile}",
        json.dumps(profile, ensure_ascii=False, indent=2),
    ).replace(
        "{extractions}",
        json.dumps(extractions, ensure_ascii=False, indent=2),
    )
    return call_llm(prompt)


# ============================================================
# 主流程
# ============================================================
def run_pipeline(data_path: str = "data/synthetic_calls.json") -> dict:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    total = len(data["customers"])

    for idx, customer in enumerate(data["customers"], 1):
        cid = customer["customer_id"]
        print(f"\n=== [{idx}/{total}] 處理 {cid} {customer['name']} ===")

        # Step 1: 抽取每通電話
        extractions = []
        for call in customer["calls"]:
            print(f"  抽取 {call['call_id']}...")
            extractions.append(extract_call(call))

        # Step 2: 建立客戶檔案
        print(f"  建立客戶檔案...")
        profile = build_profile(customer, extractions)

        # Step 3: 生成 leads
        print(f"  生成 leads...")
        leads = generate_leads(profile, extractions)

        n_leads = len(leads.get("leads", []))
        print(f"  ✅ 完成，產出 {n_leads} 個 leads")

        results[cid] = {
            "customer": customer,
            "extractions": extractions,
            "profile": profile,
            "leads": leads,
        }

    with open("output_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 全部完成，結果寫入 output_results.json")
    return results


if __name__ == "__main__":
    run_pipeline()