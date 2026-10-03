# evaluate.py
import json
import re
import feedback_store
from difflib import SequenceMatcher


def load_results(path: str = "output_results.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_evidence_validity(lead: dict, extractions: list[dict]) -> dict:
    """檢查 lead 的 supporting_evidence 是否為真實存在的 call_id"""
    valid_call_ids = {e["call_id"] for e in extractions}
    cited = set(lead.get("supporting_evidence", []))
    invalid = cited - valid_call_ids

    return {
        "lead_id": lead["lead_id"],
        "cited": list(cited),
        "invalid_citations": list(invalid),
        "has_evidence": len(cited) > 0,
        "evidence_valid": len(invalid) == 0 and len(cited) > 0,
    }


def check_rationale_grounded(lead: dict, extractions: list[dict], threshold: float = 0.3) -> dict:
    """
    檢查 lead 的 rationale 是否與 cited evidence 有語意重疊。
    用簡單的關鍵詞重疊 + 序列相似度做 proxy（POC 階段夠用）。
    """
    cited_ids = set(lead.get("supporting_evidence", []))
    cited_texts = [
        e.get("transcript", "") or json.dumps(e, ensure_ascii=False)
        for e in extractions
        if e["call_id"] in cited_ids
    ]
    combined = " ".join(cited_texts)
    rationale = lead.get("rationale", "")

    # 抽取 rationale 中的關鍵詞（2字以上中文詞 + 英文詞）
    keywords = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", rationale)
    if not keywords:
        return {"lead_id": lead["lead_id"], "overlap": 0.0, "grounded": False}

    hits = sum(1 for kw in keywords if kw in combined)
    overlap = hits / len(keywords)

    # 加上序列相似度
    seq_sim = SequenceMatcher(None, rationale, combined).ratio()
    score = max(overlap, seq_sim)

    return {
        "lead_id": lead["lead_id"],
        "keyword_overlap": round(overlap, 2),
        "seq_similarity": round(seq_sim, 2),
        "score": round(score, 2),
        "grounded": score >= threshold,
    }


def check_hallucination_indicators(lead: dict, extractions: list[dict]) -> dict:
    """
    檢查 lead 是否包含通話中沒出現的具體數字/地點（常見幻覺）。
    """
    cited_ids = set(lead.get("supporting_evidence", []))
    cited_texts = " ".join(
        json.dumps(e, ensure_ascii=False)
        for e in extractions
        if e["call_id"] in cited_ids
    )

    # 抽取 lead 中的數字與地點
    numbers = re.findall(r"\d+[\d,\.]*\s*(?:萬|億|美金|台幣|加幣|英鎊|元)?", lead.get("rationale", ""))
    suspicious = [n for n in numbers if n.strip() and n.strip() not in cited_texts]

    return {
        "lead_id": lead["lead_id"],
        "suspicious_numbers": suspicious,
        "has_hallucination_risk": len(suspicious) > 0,
    }


def evaluate_all(results: dict) -> dict:
    summary = []
    for cid, r in results.items():
        extractions = r["extractions"]
        leads = r["leads"].get("leads", [])

        for lead in leads:
            ev = check_evidence_validity(lead, extractions)
            gr = check_rationale_grounded(lead, extractions)
            ha = check_hallucination_indicators(lead, extractions)

            summary.append({
                "customer_id": cid,
                "lead_id": lead["lead_id"],
                "category": lead.get("category"),
                "priority": lead.get("priority"),
                "confidence": lead.get("confidence"),
                "evidence_valid": ev["evidence_valid"],
                "invalid_citations": ev["invalid_citations"],
                "grounded_score": gr["score"],
                "grounded": gr["grounded"],
                "hallucination_risk": ha["has_hallucination_risk"],
                "suspicious_numbers": ha["suspicious_numbers"],
            })

    # 統計
    total = len(summary)
    if total == 0:
        return {"summary": [], "stats": {}}

    stats = {
        "total_leads": total,
        "evidence_valid_rate": round(sum(1 for s in summary if s["evidence_valid"]) / total, 2),
        "grounded_rate": round(sum(1 for s in summary if s["grounded"]) / total, 2),
        "hallucination_rate": round(sum(1 for s in summary if s["hallucination_risk"]) / total, 2),
        "avg_confidence": round(sum(s["confidence"] or 0 for s in summary) / total, 2),
    }

    return {"summary": summary, "stats": stats}


def print_report(report: dict):
    print("\n" + "=" * 60)
    print("📊 NBA POC 評估報告")
    print("=" * 60)

    stats = report["stats"]
    print(f"\n總 Leads 數：{stats['total_leads']}")
    print(f"✅ Evidence 有效率：{stats['evidence_valid_rate']*100:.0f}%")
    print(f"✅ Rationale Grounded 率：{stats['grounded_rate']*100:.0f}%")
    print(f"⚠️  幻覺風險率：{stats['hallucination_rate']*100:.0f}%")
    print(f"📈 平均 Confidence：{stats['avg_confidence']}")

    print("\n--- 明細 ---")
    for s in report["summary"]:
        flag = "✅" if s["evidence_valid"] and s["grounded"] else "❌"
        print(f"{flag} [{s['customer_id']}] {s['lead_id']} {s['category']} "
              f"(P:{s['priority']}, C:{s['confidence']}) "
              f"grounded={s['grounded_score']}")
        if s["invalid_citations"]:
            print(f"     ⚠️ 無效引用：{s['invalid_citations']}")
        if s["suspicious_numbers"]:
            print(f"     ⚠️ 疑似幻覺數字：{s['suspicious_numbers']}")




def evaluate_with_feedback(results: dict) -> dict:
    """在原評估基礎上，加入 RM 回饋統計"""
    base = evaluate_all(results)
    fb = feedback_store.get_feedback_summary()

    # 按 category 統計採納率
    category_stats = {}
    for cid, r in results.items():
        for lead in r["leads"].get("leads", []):
            cat = lead.get("category", "unknown")
            fb_entry = feedback_store.get_feedback_for_lead(cid, lead["lead_id"])
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "adopt": 0}
            category_stats[cat]["total"] += 1
            if fb_entry and fb_entry["action"] == "adopt":
                category_stats[cat]["adopt"] += 1

    for cat, s in category_stats.items():
        s["adopt_rate"] = round(s["adopt"] / s["total"], 2) if s["total"] else 0.0

    return {
        "base_evaluation": base,
        "feedback_summary": fb,
        "category_stats": category_stats,
    }


def print_full_report(report: dict):
    print_report(report["base_evaluation"])

    print("\n" + "=" * 60)
    print("📈 RM 回饋統計")
    print("=" * 60)
    fb = report["feedback_summary"]
    print(f"總回饋數：{fb['total']}")
    print(f"採納：{fb['adopt']} ｜ 拒絕：{fb['reject']}")
    print(f"採納率：{fb['adopt_rate']*100:.0f}%")

    print("\n--- 各類別採納率 ---")
    for cat, s in report["category_stats"].items():
        print(f"  {cat}: {s['adopt']}/{s['total']} ({s['adopt_rate']*100:.0f}%)")

    if fb["recent_rejects"]:
        print("\n--- 最近被拒絕的 leads ---")
        for r in fb["recent_rejects"]:
            print(f"  [{r['customer_id']}] {r['lead_id']} 原因：{r.get('reason', '(未填)')}")


if __name__ == "__main__":
    results = load_results()
    report = evaluate_with_feedback(results)
    print_full_report(report)

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

