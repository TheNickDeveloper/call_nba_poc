# prompts.py
"""
Centralized LLM prompts for NBA from Call POC.

- EXTRACT_CALL_PROMPT: Extract structured info from a single call
- BUILD_PROFILE_PROMPT: Merge extractions into customer profile
- GENERATE_LEADS_PROMPT: Generate actionable leads for RM
- CRITIC_PROMPT: Quality review (LangGraph only)
- REVISE_PROMPT: Revise leads based on critic feedback (LangGraph only)
"""

# ============================================================
# Core pipeline prompts (shared by pipeline.py and graph_pipeline.py)
# ============================================================

EXTRACT_CALL_PROMPT = """You are a wealth management call analyst.
Extract structured information from the following call transcript.

【Key Principles】
1. Only extract what the customer EXPLICITLY said or STRONGLY implied. Do not speculate.
2. If the customer merely "mentioned" something without expressing a need, mark status as "mentioned" instead of "need".
3. Watch for compound scenarios: e.g., "sending kid abroad for school" may involve education fund, overseas property, FX at once.
4. Capture risk preference hints (e.g., "don't want to put everything in one place" = prefers diversification).

【Extraction Categories】
- life_events: Life events (children education, immigration, retirement, inheritance, career change)
- financial_needs: Financial needs (education fund, overseas property, asset allocation, tax planning, inheritance)
- locations: Geographic locations mentioned (UK, Canada, US, Europe)
- amounts: Amounts and currencies mentioned
- timeline: Time horizon (e.g., "sophomore in high school" -> 1-2 years, "in two or three years")
- risk_signals: Risk preference signals (quote + interpretation)
- constraints: Constraints (liquidity needs, don't want to sell house, don't touch emergency fund)
- unresolved: Open questions not yet clarified
- rm_commitments: Things the RM promised to do

【Output Format】Output JSON only, no other text:
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

Transcript:
{transcript}
"""


BUILD_PROFILE_PROMPT = """You are a wealth management customer profile analyst.
Build a customer profile from the following multi-call extraction results.

【Key Principles】
1. Recent signals weigh more (2024-09 > 2024-03).
2. If signals contradict, preserve the contradiction and annotate it. Do not force unification.
3. Tag each need with urgency (high/medium/low) and horizon (timeline).
4. Infer risk_preference from hints, but always ground it with evidence.

【Output Format】Output JSON only:
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
    {"need": "", "urgency": "high|medium|low", "horizon": "", "evidence": ["call_id"]}
  ],
  "contradictions": [
    {"signal_a": "", "signal_b": "", "note": ""}
  ]
}

Call extraction results:
{extractions}
"""


GENERATE_LEADS_PROMPT = """You are a wealth management advisor.
Generate actionable leads for the RM (Relationship Manager) based on the customer profile and call evidence.

【Key Principles】
1. Every lead MUST have supporting_evidence (specific call_id). Do not generate leads without evidence.
2. For compound scenarios, break down into independent actionable leads.
   Example: "kid studying in UK + buying property there" should be split into:
   - Education fund planning (FX, tuition cash flow)
   - Overseas property financing (mortgage, cross-border funds)
   - Asset stabilization (customer mentioned reducing stock exposure)
3. rationale should explain WHY this lead, linking to customer's risk preference and constraints.
4. suggested_action must be specific enough for the RM to know the next step.
5. Generate at most 5 leads, sorted by priority.
6. confidence reflects evidence strength: multiple calls supporting -> high; single mention -> medium.

【Output Format】Output JSON only:
{
  "customer_id": "...",
  "generated_at": "...",
  "leads": [
    {
      "lead_id": "L1",
      "category": "",
      "rationale": "",
      "suggested_action": "",
      "priority": "high|medium|low",
      "confidence": 0.0,
      "supporting_evidence": ["C001_2", "C001_3"],
      "suggested_products": [],
      "talking_points": ["First sentence the RM can say to the customer"]
    }
  ]
}

Customer Profile:
{profile}

Call Evidence:
{extractions}
"""


# ============================================================
# LangGraph-only prompts
# ============================================================

CRITIC_PROMPT = """You are a strict wealth management quality reviewer.
Review the following leads.

【Criteria - only require revision for SERIOUS issues】
Serious issues (needs_revision = true):
- supporting_evidence cites a call_id that does not exist
- rationale has NO basis in the calls (clear hallucination)
- suggested_action is completely unactionable or unclear

Acceptable (needs_revision = false):
- Wording could be better
- More details could be added
- Confidence could be tuned
- Ordering could be adjusted
- Product suggestions could be optimized

【Principle】
Prefer to let minor issues pass rather than over-request revisions.
Only return needs_revision = true when there is a truly serious issue.

Output JSON:
{
  "issues": [{"lead_id": "", "issue": "", "severity": "high|medium|low"}],
  "overall_quality": 0.0,
  "needs_revision": false,
  "revision_notes": ""
}

Leads:
{leads}

Original Call Evidence:
{extractions}
"""


REVISE_PROMPT = """You are a wealth management advisor.
Revise the following leads based on the quality reviewer's feedback.

【Review Feedback】
{review}

【Original Leads】
{leads}

【Original Call Evidence】
{extractions}

【Output Requirements - VERY IMPORTANT】
1. Only output leads that were MODIFIED. Do not repeat unchanged leads.
2. If a lead was deemed fine by the reviewer, exclude it from output.
3. If no leads need revision, return an empty array.
4. Each output lead must contain all fields (lead_id, category, rationale, suggested_action, priority, confidence, supporting_evidence, suggested_products, talking_points).

Output JSON:
{
  "revised_leads": [
    {
      "lead_id": "L1",
      "category": "...",
      "rationale": "...",
      "suggested_action": "...",
      "priority": "high|medium|low",
      "confidence": 0.0,
      "supporting_evidence": ["C001_2"],
      "suggested_products": [],
      "talking_points": []
    }
  ]
}
"""