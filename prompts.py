# prompts.py

EXTRACT_CALL_PROMPT = """你是財富管理通話分析師。請從以下逐字稿抽取結構化資訊。

【重要原則】
1. 只抽取「客戶明確表達或強烈暗示」的內容，不要臆測
2. 若客戶只是「提到」但沒表達需求，標記為 mentioned 而非 need
3. 注意複合情境：例如「送小孩出國念書」可能同時牽涉教育基金、海外置產、換匯
4. 捕捉風險偏好的蛛絲馬跡（如「不想壓在同一個地方」= 偏好分散）

【抽取類別】
- life_events: 人生事件（子女教育、移民、退休、傳承、職涯轉換）
- financial_needs: 財務需求（教育基金、海外置產、資產配置、稅務規劃、傳承）
- locations: 涉及的地理位置（英國、加拿大、美國、歐洲）
- amounts: 提到的金額與幣別
- timeline: 時間軸（如「高二」→ 1-2年內、「兩三年內」）
- risk_signals: 風險偏好訊號（原文 + 解讀）
- constraints: 限制條件（流動性需求、不想賣房、不想動預備金）
- unresolved: 尚未討論清楚的問題
- rm_commitments: RM 承諾要做的事

【輸出格式】只輸出 JSON，不要其他文字：
{
  "call_id": "...",
  "date": "...",
  "life_events": [{"type": "", "detail": "", "timeline": ""}],
  "financial_needs": [{"type": "", "detail": "", "amount": "", "status": "need|mentioned"}],
  "locations": [{"country": "", "purpose": ""}],
  "amounts": [{"value": "", "currency": "", "purpose": ""}],
  "timeline": [{"event": "", "horizon": ""}],
  "risk_signals": [{"quote": "", "interpretation": ""}],
  "constraints": [{"type": "", "detail": ""}],
  "unresolved": [],
  "rm_commitments": []
}

逐字稿：
{transcript}
"""


BUILD_PROFILE_PROMPT = """你是財富管理客戶檔案分析師。
根據以下多通通話的抽取結果，建立客戶檔案。

【重要原則】
1. 近期訊號權重較高（2024-09 比 2024-03 重要）
2. 若訊號矛盾，保留矛盾並標註，不要強行統一
3. 需求要標註 urgency（高/中/低）與 horizon（時間軸）
4. 從蛛絲馬跡推論 risk_preference，但要有 evidence

【輸出格式】只輸出 JSON：
{
  "customer_id": "...",
  "name": "...",
  "profile": {
    "life_stage": "",
    "risk_preference": "",
    "liquidity_need": "",
    "estimated_capacity": "",
    "locations_of_interest": []
  },
  "needs_timeline": [
    {"need": "", "urgency": "高|中|低", "horizon": "", "evidence": ["call_id"]}
  ],
  "contradictions": [
    {"signal_a": "", "signal_b": "", "note": ""}
  ]
}

通話抽取結果：
{extractions}
"""


GENERATE_LEADS_PROMPT = """你是財富管理顧問，要根據客戶檔案與通話證據，生成給 RM（客戶關係經理）的 leads。

【重要原則】
1. 每個 lead 必須有 supporting_evidence（具體 call_id），沒有證據的不要生成
2. 針對複合情境，要拆解成可執行的獨立 leads
   例如「送小孩去英國念書 + 當地置產」應拆成：
   - 教育基金規劃（換匯、學費現金流）
   - 海外置產融資（房貸、跨境資金安排）
   - 資產穩健化（客戶提到想減碼股票）
3. rationale 要說明「為什麼推這個」，並連結客戶的風險偏好與限制
4. suggested_action 要具體到 RM 知道下一步做什麼
5. 最多生成 5 個 leads，按 priority 排序
6. confidence 反映證據強度：多通通話支持 → 高；單一通話提及 → 中

【輸出格式】只輸出 JSON：
{
  "customer_id": "...",
  "generated_at": "...",
  "leads": [
    {
      "lead_id": "L1",
      "category": "",
      "rationale": "",
      "suggested_action": "",
      "priority": "高|中|低",
      "confidence": 0.0,
      "supporting_evidence": ["C001_2", "C001_3"],
      "suggested_products": [],
      "talking_points": ["RM 可以跟客戶說的第一句話"]
    }
  ]
}

客戶檔案：
{profile}

通話證據：
{extractions}
"""