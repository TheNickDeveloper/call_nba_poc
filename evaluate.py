# evaluate.py
"""
NBA from Call - Evaluation script.
Checks each lead for evidence validity, groundedness, hallucination risk.
Outputs evaluation_report.json for app.py sidebar.
"""
import json
import os
import re
from difflib import SequenceMatcher

import feedback_store

RESULTS_PATH = "output_results.json"
REPORT_PATH = "evaluation_report.json"


# ============================================================
# Load data
# ============================================================
def load_results(path: str = RESULTS_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python pipeline.py` or `python graph_pipeline.py` first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Check 1: Evidence validity
# ============================================================
def check_evidence_validity(lead: dict, extractions: list[dict]) -> dict:
    """Check whether the lead's supporting_evidence cites real call_ids."""
    valid_call_ids = {e["call_id"] for e in extractions if "call_id" in e}
    cited = set(lead.get("supporting_evidence", []))
    invalid = cited - valid_call_ids

    return {
        "lead_id": lead.get("lead_id"),
        "cited": list(cited),
        "invalid_citations": list(invalid),
        "has_evidence": len(cited) > 0,
        "evidence_valid": len(invalid) == 0 and len(cited) > 0,
    }


# ============================================================
# Check 2: Rationale grounded
# ============================================================
def check_rationale_grounded(
    lead: dict, extractions: list[dict], threshold: float = 0.3
) -> dict:
    """
    Check semantic overlap between lead's rationale and cited evidence.
    Uses keyword overlap + sequence similarity as a proxy (sufficient for POC).
    """
    cited_ids = set(lead.get("supporting_evidence", []))
    cited_texts = [
        json.dumps(e, ensure_ascii=False)
        for e in extractions
        if e.get("call_id") in cited_ids
    ]
    combined = " ".join(cited_texts)
    rationale = lead.get("rationale", "")

    if not rationale:
        return {
            "lead_id": lead.get("lead_id"),
            "keyword_overlap": 0.0,
            "seq_similarity": 0.0,
            "score": 0.0,
            "grounded": False,
        }

    # Extract keywords (CJK 2+ chars, English 3+ chars)
    keywords = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", rationale)
    if not keywords:
        return {
            "lead_id": lead.get("lead_id"),
            "keyword_overlap": 0.0,
            "seq_similarity": 0.0,
            "score": 0.0,
            "grounded": False,
        }

    hits = sum(1 for kw in keywords if kw in combined)
    overlap = hits / len(keywords)

    seq_sim = SequenceMatcher(None, rationale, combined).ratio()
    score = max(overlap, seq_sim)

    return {
        "lead_id": lead.get("lead_id"),
        "keyword_overlap": round(overlap, 2),
        "seq_similarity": round(seq_sim, 2),
        "score": round(score, 2),
        "grounded": score >= threshold,
    }


# ============================================================
# Check 3: Hallucination indicators
# ============================================================
def check_hallucination_indicators(lead: dict, extractions: list[dict]) -> dict:
    """
    Check whether the lead contains specific numbers not present in the calls.
    """
    cited_ids = set(lead.get("supporting_evidence", []))
    cited_texts = " ".join(
        json.dumps(e, ensure_ascii=False)
        for e in extractions
        if e.get("call_id") in cited_ids
    )

    rationale = lead.get("rationale", "")
    # Extract numbers (with optional units, both English and Chinese)
    numbers = re.findall(
        r"\d+[\d,\.]*\s*(?:萬|億|美金|台幣|加幣|英鎊|元|USD|TWD|GBP|CAD|EUR)?",
        rationale,
    )

    suspicious = [
        n.strip() for n in numbers
        if n.strip() and len(n.strip()) >= 2 and n.strip() not in cited_texts
    ]

    return {
        "lead_id": lead.get("lead_id"),
        "suspicious_numbers": suspicious,
        "has_hallucination_risk": len(suspicious) > 0,
    }


# ============================================================
# Main evaluation
# ============================================================
def evaluate_all(results: dict) -> dict:
    summary = []

    for cid, r in results.items():
        extractions = r.get("extractions", [])
        leads_container = r.get("leads", {})

        if isinstance(leads_container, dict):
            leads = leads_container.get("leads", [])
        elif isinstance(leads_container, list):
            leads = leads_container
        else:
            leads = []

        for lead in leads:
            ev = check_evidence_validity(lead, extractions)
            gr = check_rationale_grounded(lead, extractions)
            ha = check_hallucination_indicators(lead, extractions)

            summary.append({
                "customer_id": cid,
                "lead_id": lead.get("lead_id"),
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

    total = len(summary)
    if total == 0:
        return {
            "summary": [],
            "stats": {
                "total_leads": 0,
                "evidence_valid_rate": 0.0,
                "grounded_rate": 0.0,
                "hallucination_rate": 0.0,
                "avg_confidence": 0.0,
            },
        }

    stats = {
        "total_leads": total,
        "evidence_valid_rate": round(
            sum(1 for s in summary if s["evidence_valid"]) / total, 2
        ),
        "grounded_rate": round(
            sum(1 for s in summary if s["grounded"]) / total, 2
        ),
        "hallucination_rate": round(
            sum(1 for s in summary if s["hallucination_risk"]) / total, 2
        ),
        "avg_confidence": round(
            sum(s["confidence"] or 0 for s in summary) / total, 2
        ),
    }

    return {"summary": summary, "stats": stats}


# ============================================================
# Include RM feedback
# ============================================================
def evaluate_with_feedback(results: dict) -> dict:
    """Add RM feedback statistics on top of base evaluation."""
    base = evaluate_all(results)
    fb = feedback_store.get_feedback_summary()

    category_stats = {}
    for cid, r in results.items():
        leads_container = r.get("leads", {})
        if isinstance(leads_container, dict):
            leads = leads_container.get("leads", [])
        elif isinstance(leads_container, list):
            leads = leads_container
        else:
            leads = []

        for lead in leads:
            cat = lead.get("category", "unknown")
            fb_entry = feedback_store.get_feedback_for_lead(cid, lead.get("lead_id"))

            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "adopt": 0, "reject": 0}

            category_stats[cat]["total"] += 1
            if fb_entry:
                if fb_entry["action"] == "adopt":
                    category_stats[cat]["adopt"] += 1
                else:
                    category_stats[cat]["reject"] += 1

    for cat, s in category_stats.items():
        s["adopt_rate"] = round(s["adopt"] / s["total"], 2) if s["total"] else 0.0

    return {
        "base_evaluation": base,
        "feedback_summary": fb,
        "category_stats": category_stats,
    }


# ============================================================
# Reporting
# ============================================================
def print_report(report: dict):
    print("\n" + "=" * 60)
    print("NBA POC Evaluation Report")
    print("=" * 60)

    stats = report["stats"]
    print(f"\nTotal leads: {stats['total_leads']}")
    print(f"Evidence valid rate: {stats['evidence_valid_rate'] * 100:.0f}%")
    print(f"Rationale grounded rate: {stats['grounded_rate'] * 100:.0f}%")
    print(f"Hallucination risk rate: {stats['hallucination_rate'] * 100:.0f}%")
    print(f"Average confidence: {stats['avg_confidence']}")

    print("\n--- Details ---")
    for s in report["summary"]:
        flag = "[OK]" if s["evidence_valid"] and s["grounded"] else "[!!]"
        print(
            f"{flag} [{s['customer_id']}] {s['lead_id']} {s['category']} "
            f"(P:{s['priority']}, C:{s['confidence']}) "
            f"grounded={s['grounded_score']}"
        )
        if s["invalid_citations"]:
            print(f"      [WARN] Invalid citations: {s['invalid_citations']}")
        if s["suspicious_numbers"]:
            print(f"      [WARN] Suspicious numbers: {s['suspicious_numbers']}")


def print_full_report(report: dict):
    print_report(report["base_evaluation"])

    print("\n" + "=" * 60)
    print("RM Feedback Summary")
    print("=" * 60)
    fb = report["feedback_summary"]
    print(f"Total feedbacks: {fb['total']}")
    print(f"Adopted: {fb['adopt']} | Rejected: {fb['reject']}")
    print(f"Adoption rate: {fb['adopt_rate'] * 100:.0f}%")

    if report["category_stats"]:
        print("\n--- Adoption rate by category ---")
        for cat, s in report["category_stats"].items():
            print(
                f"  {cat}: {s['adopt']}/{s['total']} "
                f"({s['adopt_rate'] * 100:.0f}%)"
            )

    if fb.get("recent_rejects"):
        print("\n--- Recent rejected leads ---")
        for r in fb["recent_rejects"]:
            print(
                f"  [{r['customer_id']}] {r['lead_id']} "
                f"reason: {r.get('reason', '(none)')}"
            )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    try:
        results = load_results()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    report = evaluate_with_feedback(results)
    print_full_report(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Evaluation report written to {REPORT_PATH}")