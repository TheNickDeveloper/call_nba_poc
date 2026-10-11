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
from confidence import expected_priority

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
# Magnitude suffixes, so "7M", "7,000,000" and "7 million" compare equal.
_UNIT_MULT = {"K": 10 ** 3, "M": 10 ** 6, "B": 10 ** 9, "萬": 10 ** 4, "億": 10 ** 8}
_NUMBER_RE = re.compile(r"(\d+(?:[,.]\d+)*)\s*(萬|億|[KMB])?")

# Identifier tokens (L1, Q4, C004_1, CUST_005) are labels, not amounts. The
# pattern needs a letter first, so "7M", "USD 800,000" and "2024-09" are safe.
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z]{1,4}[-_]?\d{1,4}(?:[-_]\d{1,4})?\b")


def _strip_identifiers(text: str) -> str:
    return _IDENTIFIER_RE.sub(" ", text)


def _number_magnitudes(text: str) -> set:
    """Normalise every number in `text` to a comparable magnitude."""
    out = set()
    for raw_value, unit in _NUMBER_RE.findall(text):
        try:
            value = float(raw_value.replace(",", ""))
        except ValueError:
            continue
        if unit:
            value *= _UNIT_MULT[unit]
        out.add(round(value))
    return out

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

    # Compare normalised magnitudes: a formatting difference is not a hallucination.
    # Identifier tokens (L1, Q4, C004_1) are removed first so they are not
    # mistaken for amounts.
    evidence_numbers = _number_magnitudes(_strip_identifiers(cited_texts))
    suspicious = [
        f"{value:,}"
        for value in sorted(
            _number_magnitudes(_strip_identifiers(rationale)) - evidence_numbers
        )
    ]

    return {
        "lead_id": lead.get("lead_id"),
        "suspicious_numbers": suspicious,
        "has_hallucination_risk": len(suspicious) > 0,
    }


# ============================================================
# Check 4: English output compliance
# ============================================================
# talking_points stay in the customer's language by design, so they are exempt.
_HAN = re.compile(r"[\u4e00-\u9fff]")


def check_english_compliance(lead: dict) -> dict:
    """Check that the lead's analysis fields contain no CJK characters."""
    texts = [
        lead.get("category") or "",
        lead.get("rationale") or "",
        lead.get("suggested_action") or "",
    ]
    texts += [p for p in (lead.get("suggested_products") or []) if isinstance(p, str)]

    non_english = [t for t in texts if _HAN.search(t)]
    return {
        "lead_id": lead.get("lead_id"),
        "english_ok": not non_english,
        "non_english_fields": len(non_english),
    }


# ============================================================
# Check 5: confidence / priority consistency
# ============================================================
def check_priority_consistency(lead: dict) -> dict:
    """Priority must follow from confidence under the published bands."""
    conf = lead.get("confidence")
    expected = expected_priority(conf) if isinstance(conf, (int, float)) else None
    return {
        "lead_id": lead.get("lead_id"),
        "expected_priority": expected,
        "priority_consistent": expected is not None and lead.get("priority") == expected,
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
            en = check_english_compliance(lead)
            pc = check_priority_consistency(lead)

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
                "english_ok": en["english_ok"],
                "priority_consistent": pc["priority_consistent"],
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
                "language_compliance_rate": 0.0,
                "priority_consistency_rate": 0.0,
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
        "language_compliance_rate": round(
            sum(1 for s in summary if s["english_ok"]) / total, 2
        ),
        "priority_consistency_rate": round(
            sum(1 for s in summary if s["priority_consistent"]) / total, 2
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
    print(f"English compliance rate: {stats.get('language_compliance_rate', 0) * 100:.0f}%")
    print(f"Priority consistency rate: {stats.get('priority_consistency_rate', 0) * 100:.0f}%")

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