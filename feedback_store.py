# feedback_store.py
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
    記錄 RM 對某個 lead 的回饋。
    action: adopt（採納）或 reject（不採納）
    """
    assert action in ("adopt", "reject"), "action 必須是 adopt 或 reject"

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
    """取得某 lead 的最新回饋（若重複回饋取最新一筆）"""
    data = _load()
    matches = [
        f for f in data["feedbacks"]
        if f["customer_id"] == customer_id and f["lead_id"] == lead_id
    ]
    return matches[-1] if matches else None


def get_feedback_summary() -> dict:
    """統計所有回饋"""
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

    # 依 category 統計（需要外部傳 category 進來，這裡先留空）
    recent_rejects = [
        f for f in feedbacks[-20:] if f["action"] == "reject"
    ]

    return {
        "total": len(feedbacks),
        "adopt": adopt,
        "reject": reject,
        "adopt_rate": round(adopt / len(feedbacks), 2),
        "recent_rejects": recent_rejects,
    }


def get_adopted_examples(limit: int = 10) -> list[dict]:
    """取得最近被採納的 leads，可用於 few-shot 學習"""
    data = _load()
    adopted = [f for f in data["feedbacks"] if f["action"] == "adopt"]
    return adopted[-limit:]


def get_rejected_examples(limit: int = 10) -> list[dict]:
    """取得最近被拒絕的 leads，可用於避免重複生成"""
    data = _load()
    rejected = [f for f in data["feedbacks"] if f["action"] == "reject"]
    return rejected[-limit:]

def remove_feedback(customer_id: str, lead_id: str) -> bool:
    """
    移除某 lead 的所有回饋紀錄（用於撤銷後重新回饋）。
    回傳 True 表示有移除任何紀錄。
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