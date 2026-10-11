# -*- coding: utf-8 -*-
"""Deterministic confidence & priority for NBA leads.

Replaces the LLM's self-asserted `confidence` (which never went below 0.55 and
contradicted the `priority` field) with a value computed from evidence.

The LLM's own estimate is preserved as `confidence_llm`, and its priority as
`priority_llm`, so the two can be compared.

Scoring is kept in this module and imported by BOTH graph_pipeline.py and
evaluate.py, so the bands cannot drift apart.
"""

# --- scoring rules -----------------------------------------------------------
EVIDENCE_BASE = {
    0: 0.30,   # no valid citation
    1: 0.55,   # single-call evidence
    2: 0.75,   # corroborated by two calls
}
EVIDENCE_MAX = 0.90          # three or more distinct evidence calls
RECENCY_BONUS = 0.05         # cited set covers the customer's most recent call
MENTIONED_PENALTY = -0.10    # cited calls contain no confirmed "need" signal
FLOOR, CEIL = 0.30, 0.95

# --- priority bands ----------------------------------------------------------
HIGH_BAND = 0.80
MEDIUM_BAND = 0.65

STEP = 0.05


def expected_priority(confidence: float) -> str:
    """The priority implied by a confidence score."""
    if confidence >= HIGH_BAND:
        return "high"
    if confidence >= MEDIUM_BAND:
        return "medium"
    return "low"


def latest_call_id(customer: dict) -> str | None:
    """The most recent call for a customer, by date."""
    calls = (customer or {}).get("calls") or []
    if not calls:
        return None
    return max(calls, key=lambda c: c.get("date", ""))["call_id"]


def compute_confidence(lead: dict, extractions: list, latest_id: str | None = None) -> float:
    """Confidence from evidence count, recency and signal status.

    Only citations that exist in `extractions` count, so a fabricated call_id
    cannot raise the score.
    """
    valid_ids = {e.get("call_id") for e in extractions if e.get("call_id")}
    cited = {c for c in (lead.get("supporting_evidence") or []) if c in valid_ids}
    n = len(cited)

    score = EVIDENCE_MAX if n >= 3 else EVIDENCE_BASE.get(n, EVIDENCE_MAX)

    if latest_id and latest_id in cited:
        score += RECENCY_BONUS

    cited_extractions = [e for e in extractions if e.get("call_id") in cited]
    has_confirmed_need = any(
        need.get("status") == "need"
        for e in cited_extractions
        for need in (e.get("financial_needs") or [])
    )
    if cited_extractions and not has_confirmed_need:
        score += MENTIONED_PENALTY

    score = min(max(score, FLOOR), CEIL)
    # Snap to STEP and round away binary float noise (0.05 * 6 = 0.30000000000000004).
    return round(round(score / STEP) * STEP, 2)


def apply_computed_confidence(leads, extractions: list, latest_id: str | None = None) -> None:
    """Overwrite confidence/priority on every lead in place.

    Idempotent: the LLM's original values are captured only once.
    """
    if not isinstance(leads, list):
        return

    for lead in leads:
        if not isinstance(lead, dict):
            continue
        if "confidence_llm" not in lead:
            lead["confidence_llm"] = lead.get("confidence")
        if "priority_llm" not in lead:
            lead["priority_llm"] = lead.get("priority")

        score = compute_confidence(lead, extractions, latest_id)
        lead["confidence"] = score
        lead["priority"] = expected_priority(score)
