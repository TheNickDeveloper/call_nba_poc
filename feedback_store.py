# feedback_store.py
"""
Persist RM feedback (adopt / reject) into feedback.json.
Used by graph_pipeline.py (feedback loop) and app.py (UI).
"""
import json
import os
from datetime import datetime

FEEDBACK_PATH = "feedback.json"


def _load() -> dict:
    if not os.path.exists(FEEDBACK_PATH):
        return {"feedbacks": []}
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict):
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_feedback(
    customer_id: str,
    lead_id: str,
    action: str,  # "adopt" | "reject"
    reason: str = "",
    rm_name: str = "unknown",
) -> dict:
    """
    Record RM feedback for a lead.
    action: "adopt" or "reject"
    """
    assert action in ("adopt", "reject"), "action must be 'adopt' or 'reject'"

    data = _load()
    entry = {
        "customer_id": customer_id,
        "lead_id": lead_id,
        "action": action,
        "reason": reason,
        "rm_name": rm_name,
        "timestamp": datetime.now().isoformat(),
    }
    data["feedbacks"].append(entry)
    _save(data)
    return entry


def get_feedback_for_lead(customer_id: str, lead_id: str) -> dict | None:
    """Get latest feedback for a specific lead (latest if duplicated)."""
    data = _load()
    matches = [
        f for f in data["feedbacks"]
        if f["customer_id"] == customer_id and f["lead_id"] == lead_id
    ]
    return matches[-1] if matches else None


def get_feedback_summary() -> dict:
    """Aggregate all feedback."""
    data = _load()
    feedbacks = data["feedbacks"]

    if not feedbacks:
        return {
            "total": 0,
            "adopt": 0,
            "reject": 0,
            "adopt_rate": 0.0,
            "by_category": {},
            "recent_rejects": [],
        }

    adopt = sum(1 for f in feedbacks if f["action"] == "adopt")
    reject = len(feedbacks) - adopt

    recent_rejects = [
        f for f in feedbacks[-20:] if f["action"] == "reject"
    ]

    return {
        "total": len(feedbacks),
        "adopt": adopt,
        "reject": reject,
        "adopt_rate": round(adopt / len(feedbacks), 2),
        "by_category": {},
        "recent_rejects": recent_rejects,
    }


def get_adopted_examples(limit: int = 10) -> list[dict]:
    """Get recently adopted leads (for few-shot learning)."""
    data = _load()
    adopted = [f for f in data["feedbacks"] if f["action"] == "adopt"]
    return adopted[-limit:]


def get_rejected_examples(limit: int = 10) -> list[dict]:
    """Get recently rejected leads (to avoid repetition)."""
    data = _load()
    rejected = [f for f in data["feedbacks"] if f["action"] == "reject"]
    return rejected[-limit:]


def remove_feedback(customer_id: str, lead_id: str) -> bool:
    """
    Remove all feedback for a lead (for undo + re-feedback).
    Returns True if any record was removed.
    """
    data = _load()
    before = len(data["feedbacks"])
    data["feedbacks"] = [
        f for f in data["feedbacks"]
        if not (f["customer_id"] == customer_id and f["lead_id"] == lead_id)
    ]
    after = len(data["feedbacks"])
    if after < before:
        _save(data)
        return True
    return False