# Next Best Action from Call — Multi-Call Reasoning + Agentic Lead Generation on LangGraph

> An agentic system that reads RM-customer call transcripts, infers customer needs across time, and generates prioritized, evidence-backed next best actions — with a closed feedback loop from RMs.
>
> This document breaks down how it works from a workflow perspective, and what value it delivers to the business.

---

## Application UIUX

![image](https://github.com/TheNickDeveloper/call_nba_poc/blob/main/application_uiux.png)

---

## 1. What Problem This POC Solves

Wealth management RMs sit on a goldmine of customer intelligence — but it's trapped in call recordings and scattered notes. Every conversation reveals a fragment: a child's education plan, a property purchase intention, a shift in risk appetite. Individually, these fragments are forgettable. Collectively, they form a customer's life trajectory.

The traditional "CRM + manual notes" approach has three fatal flaws:

- **Fragmented**: Signals from March, June, and September calls are never stitched into a coherent picture. The RM relies on memory or Ctrl+F.
- **Reactive**: The RM waits for the customer to ask. By the time the customer says "I want to buy a UK property," competitors are already pitching.
- **Inconsistent**: A senior RM spots the "child studying abroad → overseas property → FX planning" chain. A junior RM misses it. Customer experience varies by who answers the phone.

This POC uses a **Multi-Call Reasoning + Agentic Lead Generation** architecture to solve all three simultaneously.

---

## 2. End-to-End Workflow

The system operates in two phases: **offline lead generation** and **online RM consumption with feedback loop**.

### Phase 1: Offline Lead Generation (Batch)

```text
Call Transcripts → Per-Call Extraction → Cross-Call Profile → Lead Generation → Critic Review → Validated Leads
```

**Step 1 | Call Transcripts**

Input is a set of call transcripts (synthetic for POC; production would ingest from ASR or CRM). Each customer has 3-5 calls spanning months, mimicking real RM-customer relationships.

**Step 2 | Per-Call Extraction**

Each call is independently processed by an LLM to extract structured signals: life events, financial needs, risk preferences, constraints, unresolved questions, and RM commitments. This converts unstructured dialogue into machine-readable facts.

**Step 3 | Cross-Call Profile Building**

All extractions for a customer are merged into a single profile. The LLM applies **temporal weighting** (recent signals matter more), **contradiction preservation** (if the customer said "conservative" in March but "willing to try high-risk" in September, both are kept and flagged), and **needs timeline construction** (what does the customer need, when, and how urgently).

**Step 4 | Lead Generation**

Given the profile and evidence, the LLM generates 3-5 actionable leads. Each lead includes rationale, suggested action, priority, confidence, evidence citations, product recommendations, and RM talking points.

**Step 5 | Critic Review (Agentic Self-Correction)**

A second LLM acts as a quality reviewer. It checks: Does every lead cite real evidence? Is the rationale grounded in the calls, or hallucinated? Is the suggested action specific enough? If serious issues are found, the system **automatically revises** the leads — up to 3 iterations.

**Output**: For each customer, a prioritized, evidence-backed set of next best actions.

---

### Phase 2: Online RM Consumption + Feedback Loop (Per Session)

```text
RM Opens Dashboard → Selects Customer → Reviews Leads → Adopts/Rejects → Feedback Stored → Influences Next Generation
```

#### Orchestration: LangGraph StateGraph

The lead generation flow is orchestrated with **[LangGraph](https://github.com/langchain-ai/langgraph)**, not a plain sequential script. `graph_pipeline.py` defines a `StateGraph` whose nodes share a typed `NBAState`:

```text
START → extract → profile → load_feedback → generate → critic → (revise ↺ critic)* → finalize → END
```

| Node | Responsibility |
| --- | --- |
| `extract` | Per-call LLM extraction into structured JSON |
| `profile` | Cross-call merge, temporal weighting, contradiction detection |
| `load_feedback` | Load RM's past adopt/reject history for personalization |
| `generate` | Produce leads with rationale, evidence, actions |
| `critic` | Quality review — catch hallucinations, weak evidence |
| `revise` | Fix only modified leads, merging back into the set |
| `finalize` | Emit the final validated leads |

How the steps below map to graph nodes:

| Step | Graph node |
| --- | --- |
| Step 2 — Per-Call Extraction | `extract` |
| Step 3 — Cross-Call Profile | `profile` |
| Step 3.5 — Feedback Loading | `load_feedback` |
| Step 4 — Lead Generation | `generate` |
| Step 5 — Critic Review | `critic` → `revise` (conditional) |

Why LangGraph instead of a linear chain:

- **Explicit state** — every intermediate value (`extractions`, `profile`, `leads`, `review_notes`, `final_leads`) lives in `NBAState`, so the data flow is inspectable and debuggable.
- **Self-correcting** — the `critic → revise ↺ critic` loop catches hallucinations before they reach the RM. This is not possible with a one-pass chain.
- **Feedback-aware** — the `load_feedback` node injects RM's past decisions into the generation prompt, enabling personalization at scale.
- **Graceful fallback** — if a customer fails (API timeout, JSON truncation), the try/except wraps each customer independently; the pipeline completes for the rest.
- **Extensible by design** — any linear edge can be swapped for a conditional edge to add compliance checks, product matching, or human-in-the-loop approval without rewriting the nodes.
- **Visualizable** — the compiled graph can be exported as a PNG directly from the code for stakeholder review.

The graph is compiled once in `run_graph_pipeline` and reused across all customers.

#### Step-by-step

**Step 6 | RM Opens Dashboard**

Streamlit app loads `output_results.json` and `evaluation_report.json`, rendering the sidebar with system metrics (evidence valid rate, grounded rate, hallucination risk) and RM feedback stats (adoption rate).

**Step 7 | RM Selects Customer — Streamlit sidebar**

The sidebar lists all customers. The main panel shows the customer profile:

- Life stage, risk preference, liquidity need, estimated capacity, locations of interest
- Needs timeline (what, when, urgency, evidence)
- Contradictions (flagged if any)
- Critic review score and issues found

**Step 8 | RM Reviews Leads — Priority-sorted cards**

Leads are sorted by priority (high → medium → low). Each lead card expands to show:

- **Rationale** — why this lead was generated, linked to customer's risk profile and constraints
- **Suggested Action** — specific next step for the RM
- **RM Talking Points** — opening lines the RM can use
- **Evidence** — specific call IDs that triggered this lead
- **Suggested Products** — relevant financial products
- **Confidence bar** — visual indicator of evidence strength

**Step 9 | RM Adopts or Rejects — Feedback capture**

Two buttons per lead:

- **Adopt** — one click, recorded immediately
- **Reject** — opens a reason input field, then records

Every decision is stored in `feedback.json` with RM name and timestamp. RM can undo and re-feedback.

**Step 10 | Feedback Persists and Influences Next Run**

On the next `python graph_pipeline.py` run, the `load_feedback` node reads adopted/rejected examples and injects them into the generation prompt:

- **Adopted examples** → "generate in similar style"
- **Rejected examples** → "avoid similar patterns, or strengthen evidence"

This closes the loop: the system learns RM preferences without any model fine-tuning.

**Step 11 | Evaluation Report Updates**

`python evaluate.py` recalculates:

- Evidence validity rate (do leads cite real call IDs?)
- Rationale groundedness (is the rationale supported by evidence?)
- Hallucination risk (are there numbers not in the calls?)
- Adoption rate by category (which lead types do RMs trust?)

**Step 12 | Return to RM**

The updated dashboard reflects the latest leads, feedback stats, and quality metrics — giving RMs and managers a real-time view of system performance.

---

## 3. Complete Data Flow

### Offline Lead Generation — LangGraph StateGraph

```text
Call Transcripts (data/synthetic_calls.json)
    ↓
[extract] LLM extraction per call → structured JSON
    ↓
[profile] Cross-call merge → customer profile + timeline
    ↓
[load_feedback] RM feedback history (feedback.json)
    ↓
[generate] LLM lead generation with rationale + evidence
    ↓
[critic] Quality review (hallucination, evidence validity)
    ↓
[revise] Self-correction if serious issues found (up to 3 iterations)
    ↓
[finalize] output_results.json
```

### Online RM Consumption — Streamlit

```text
RM Dashboard (app.py)
    ↓
Select Customer → View Profile + Leads
    ↓
Adopt / Reject per Lead → feedback.json
    ↓
Evaluation Report (evaluate.py) → evaluation_report.json
```

### Closed Loop

```text
feedback.json → next graph_pipeline run → [load_feedback] node
```

---

## 4. Business Value

### 1. Turns Fragmented Conversations into a Coherent Customer Picture

A single call is noise. Three calls across six months is signal. The system extracts life events, financial needs, and risk signals from each call, then merges them with **temporal weighting** — so a September shift in risk appetite overrides a March baseline. Contradictions are preserved and flagged, not silently averaged away.

**Business value**: RMs no longer need to remember or re-read call notes. The system surfaces "what this customer actually needs, right now."

### 2. Proactive Lead Generation Instead of Reactive Pitching

The system doesn't wait for the customer to ask. It infers the next need from the pattern — "child studying abroad in 18 months" → "tuition cash flow planning" → "overseas property financing" → "cross-border FX strategy." Each step is a separate, actionable lead.

**Business value**: RM transitions from order-taker to trusted advisor. The customer feels understood, not sold to.

### 3. Every Lead Is Traceable and Verifiable

Each lead cites specific call IDs (e.g., `C001_2`, `C001_3`). The RM can expand the evidence and see exactly which conversation triggered this recommendation. `evaluate.py` automatically verifies that every cited call ID actually exists in the extraction set.

**Business value**: Meets compliance and audit requirements in financial services. Recommendations aren't "the AI said so" — they're "according to call C001_2 and C001_3."

### 4. Self-Correcting Quality via Critic Agent

A second LLM reviews every lead for hallucinations, invalid evidence citations, and vague actions. If serious issues are found, the system revises automatically — up to 3 iterations. The Critic is designed to **prefer letting minor issues pass** rather than over-requesting revisions, which keeps costs low.

**Business value**: In finance, a fabricated number is more dangerous than no recommendation. This design turns "untrustworthy AI output" into "auditable, self-correcting leads."

### 5. Honest Boundaries, Not Fabricated Confidence

Every lead carries a `confidence` score reflecting evidence strength: multiple calls supporting → high; single mention → medium. Leads without evidence are not generated at all. The evaluation report quantifies hallucination risk so the team knows exactly how trustworthy the output is.

**Business value**: RMs can trust the priority ordering. A "high" lead actually means "high evidence," not "the AI felt confident."

### 6. Different Customers, Different Leads — Automatically

A 48-year-old tech executive planning UK education gets education fund + overseas property + asset stabilization leads. A 62-year-old business owner planning succession gets cross-border tax + trust structuring + stable income leads. A 42-year-old restaurateur planning Canada immigration gets cross-border asset allocation leads.

**Business value**: One system serves diverse customer segments without manual segmentation or rule maintenance.

### 7. RM Feedback Loop Without Model Fine-Tuning

Every Adopt/Reject is stored and injected into the next generation prompt:

- **Adopted examples** → "generate in similar style"
- **Rejected examples** → "avoid similar patterns, or strengthen evidence"

If RMs consistently reject "privacy asset holding" leads, the system learns to avoid them — or strengthen the evidence. No retraining, no expensive fine-tuning.

**Business value**: The system improves with use, without expensive retraining. RM preferences become institutional knowledge.

### 8. Zero Marginal Cost for RM Consumption

Lead generation requires API calls, but once `output_results.json` is produced, the RM dashboard reads from a local file. No API cost per RM session.

**Business value**: Scales to dozens of RMs with negligible cost. Only pipeline runs incur API expenses — estimated **~$0.01 for 5 customers** with DeepSeek's pricing.

### 9. Quantitative Quality Measurement Built In

`evaluate.py` computes four key metrics automatically:

| Metric | What It Measures |
| --- | --- |
| Evidence valid rate | % of leads citing real call IDs |
| Rationale grounded rate | % of leads whose rationale overlaps with evidence |
| Hallucination risk rate | % of leads containing numbers not in calls |
| Adoption rate by category | Which lead types RMs trust most |

**Business value**: The team can track quality over time, A/B test prompt changes, and prove ROI to stakeholders with numbers, not anecdotes.

### 10. Extensible into a Full Wealth Advisory Platform

The current architecture scales horizontally:

- Add **ASR** to ingest real call recordings automatically
- Add **product catalog matching** to suggest specific funds/insurance
- Add **compliance checks** (suitability, KYC) as additional LangGraph nodes
- Connect to **CRM** for real-time customer data
- Add **multi-language support** for cross-border customers
- Add **human-in-the-loop** review for high-priority leads

**Business value**: Evolves from a POC into production-grade advisory infrastructure without architectural rewrite.

---

## 5. Tech Stack Summary

| Layer | Technology | Role |
| --- | --- | --- |
| Frontend | Streamlit | RM dashboard: customer profile + leads + feedback |
| Orchestration | LangGraph (StateGraph) | Agent workflow: `extract → profile → load_feedback → generate → critic → revise ↺ → finalize` |
| LLM | DeepSeek (OpenAI-compatible) | Extraction, profile building, lead generation, critic review |
| Feedback | JSON file (`feedback.json`) | RM adopt/reject log, injected into next generation |
| Evaluation | Custom Python (`evaluate.py`) | Evidence validity, groundedness, hallucination detection |
| Constraints | System prompts | Prevent hallucination, enforce evidence citations |
| Data | Synthetic call transcripts | POC input; production would ingest from ASR/CRM |

---

## 6. Comparison: This POC vs Traditional CRM

| Dimension | Traditional CRM | This POC |
| --- | --- | --- |
| **Data storage** | Manual notes, tags | Structured extraction from transcripts |
| **Cross-call reasoning** | RM's memory | LLM merges across time, weights recency |
| **Lead generation** | Manual, rule-based | LLM-generated, evidence-backed, prioritized |
| **Quality control** | Manager review (post-hoc) | Critic agent (real-time self-correction) |
| **Personalization** | Static segments | Learns from RM feedback loop |
| **Auditability** | "RM said so" | Every lead cites specific call IDs |
| **Scalability** | Linear with headcount | One pipeline serves all RMs, all customers |
| **Improvement** | Retraining, process redesign | Feedback loop improves output without retraining |

---

## 7. What Success Looks Like

If this POC moves to production, success would be measured by:

| Metric | Target |
| --- | --- |
| RM adoption rate of leads | > 60% within 3 months |
| Lead generation time per customer | < 60 seconds (vs. 15+ minutes manual) |
| Evidence validity rate | > 95% |
| Hallucination rate | < 5% |
| Cross-sell conversion lift | +20% vs. baseline |
| RM time saved per week | 5+ hours |

---

## 8. One-Sentence Summary

> The core value of this POC isn't "letting an LLM read call transcripts" — it's **turning every customer conversation into a proactive, evidence-backed next best action, and letting RMs teach the system what good looks like over time**. That's what wealth management actually needs from AI.
>
> And it does so with an **explicit LangGraph workflow** — every step (extraction, profiling, feedback loading, generation, critic review, revision) is a named node in a state graph, so the logic is inspectable, testable, and ready to grow into compliance checks, product matching, or human-in-the-loop approval.

---

**This document is designed to be read by business stakeholders, product owners, and compliance teams — not just engineers. The technical details live in the code; the business logic lives here.**