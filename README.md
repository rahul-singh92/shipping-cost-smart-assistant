# Shipping Cost Smart Assistant

An automated, reproducible freight cost auditing pipeline and Retrieval-Augmented Generation (RAG) assistant. It analyzes historical shipment records to detect route-level weekly cost spikes and quiet cost creep, cross-references flagged anomalies against operational context notes using an open-source LLM (`qwen/qwen3.8-27b` via Groq), and separates **justified disruptions** from **unexplained cost hikes** requiring human review.

---

## 1. System Architecture & Pipeline Design

The system is strictly decoupled into deterministic quantitative analysis (Pandas/NumPy) and grounded qualitative reasoning (RAG with JSON schema enforcement):

    Shipping Cost Smart Assistant/
    ├── data/
    │   ├── shipment_records.csv          # Raw shipment-level transaction logs
    │   └── context_notes.csv             # Operational & market event logs (N001-N010)
    ├── src/
    │   ├── data_pipeline.py              # Weekly aggregation & baseline computation
    │   ├── anomaly_detector.py           # Two-part anomaly detection & formatting
    │   └── rag_engine.py                 # Candidate retrieval, prompt guardrails & Groq LLM client
    ├── output/
    │   ├── shipping_cost_anomalies.csv   # Final submission output matching sample contract
    │   ├── run_1_output.csv              # Reproducibility Run 1
    │   ├── run_2_output.csv              # Reproducibility Run 2
    │   ├── run_3_output.csv              # Reproducibility Run 3
    │   └── run_1_cost_log.json           # Token usage and cost audit log
    ├── main.py                           # End-to-end pipeline orchestrator
    ├── evaluate.py                       # Ground-truth eval harness & 3-run reproducibility verifier
    ├── ask_assistant.py                  # Interactive plain-English Q&A CLI (Stretch Goal)
    └── requirements.txt                  # Python dependencies

---

## 2. Mathematical Baselines & Edge-Case Handling

### Weekly Cost per Tonne-km
All shipment dates are mapped to their Monday-start week (`week_of`). Rather than averaging individual shipment ratios (which biases toward small loads), the weekly route metric is computed as the true volume-weighted ratio:

    Weekly Cost per Tonne-km = Sum(Freight Cost INR) / Sum(Quantity Tonnes * Distance km)

### Baseline 1: Own Historical Baseline (Trailing 8-Week Window)
- For each route and week `t`, the historical baseline is the rolling mean of the preceding 8 weeks (`t-8` through `t-1`), implemented via `.shift(1).rolling(window=8, min_periods=1).mean()`.
- **No Look-Ahead Guarantee:** Shifting by 1 week ensures the target week `t` is strictly excluded from its own baseline.
- **Handling Routes with Fewer Than 8 Prior Weeks:**
  - **Week 1 of a route (0 prior weeks):** `historical_baseline` evaluates to `NaN` (`min_periods=1` after `.shift(1)`). The route is not falsely flagged on history, and can still be evaluated against peer routes.
  - **Weeks 2 to 8 of a route (1 to 7 prior weeks):** Computed as the partial trailing average of all available prior weeks (`min_periods=1`). This prevents cold-start blindness during the first two months of data while preserving the strict no-look-ahead boundary.

### Baseline 2: Peer Route Baseline (Same Route Type & Week)
- Compares a route's weekly cost per tonne-km against all *other* routes sharing the same `route_type` (`Short`, `Medium`, `Long`) during the exact same `week_of`:

    Peer Baseline = (Sum of all routes of same type in week t - Own Cost) / (Count of routes of same type - 1)

- If a route is the sole active route of its type in a given week (`N <= 1`), `peer_baseline` safely defaults to `NaN`.

---

## 3. Anomaly Detection Logic & Threshold Trade-offs

A naive single threshold (e.g., `>5%` on either baseline) flags 219 out of 728 route-weeks because certain routes (`Mumbai-Pune`) have structurally higher base rates than their peers (`Delhi-Jaipur`) even when prices are flat. Conversely, an overly high historical threshold (`>15%`) misses nationwide fuel surcharges (`+5% to +7%`) and gradual multi-week cost creep.

To capture both failure modes cleanly, `src/anomaly_detector.py` implements a **two-part rule** (flagging 24 genuine route-weeks out of 728):

1. **Direct Historical Spike (`pct_vs_history >= 7.0%`):** Captures sudden route-specific surges (`Delhi-Jaipur` +35.5%, `Chennai-Bangalore` +31.5%, `Ahmedabad-Mumbai` +29.5%) as well as the May 2025 nationwide diesel hike (`+7.3%` to `+11.6%` vs trailing history while peer difference stays near 0% because all peers rose simultaneously).
2. **Quiet Cost Creep (`pct_vs_history >= 5.0%` AND `pct_vs_peers >= 20.0%`):** Captures routes that are actively rising against their own trailing window *and* diverging sharply from peer corridors (such as `Mumbai-Pune` in late 2025, matching `2025-09-15` in the reference output).

---

## 4. Hallucination Guardrails & Temporal RAG Filtering

1. **Deterministic Pre-Filtering (Zero-Token Fast Path):** `find_candidate_notes()` filters `context_notes.csv` by route (`Origin-Destination` or `All Routes`) and a temporal window (`[week_of - 28 days, week_of + 7 days]`). If 0 candidate notes exist (8 of the 24 anomalies), the pipeline bypasses the LLM call completely and deterministically returns `flagged="Yes"`, `matched_note_id=""`.
2. **Strict Citation Whitelisting:** When candidate notes exist, the prompt supplies an explicit list of allowed IDs and enforces `response_format={"type": "json_object"}` with `temperature=0.0` and `seed=42`. Post-processing validates the returned `note_id` against the candidate whitelist and forces `matched_note_id=""` whenever `verdict == "Yes"`.
3. **Temporal Boundary & Distractor Rejection:**
   - **Temporal Expiration (`N001` & `N009`):** `Chennai-Bangalore` spikes on `2025-02-24` and `2025-03-03` are justified by `N001` (flooding Feb 24–Mar 8). However, spikes on `2025-03-10` and `2025-03-17` are flagged as **`Yes` (unexplained)** because `N001` explicitly notes roads were passable by Mar 9 and `N009` confirms normal conditions.
   - **Distractor Notes (`N006` & `N010`):** `Mumbai-Pune` spikes in Sep–Nov 2025 retrieve `N006` ("stable demand, no major disruptions") and `N010` ("compliance costs absorbed without a rate change"). The RAG engine rejects both as non-justifying and flags the rows as **`Yes`** with a blank `matched_note_id`.

---

## 5. Evaluation Harness & 3-Run Reproducibility Verification

Running `python evaluate.py` executes the full pipeline **3 times from scratch** (`output/run_1_output.csv`, `output/run_2_output.csv`, `output/run_3_output.csv`) and validates:

- **Check 1 (Schema Contract):** Exact match against `route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id,reason`.
- **Check 2 (Citation Invariants):** 0 instances of `flagged="Yes"` with a non-empty `matched_note_id`, and 0 instances of `flagged="No (justified)"` with a blank `matched_note_id`.
- **Check 3 (10-Case Ground-Truth Eval Set):** **10/10 (100.0%)** accuracy across sudden unexplained spikes, festival surcharges (`N002`), active vs. expired flooding (`N001`/`N009`), nationwide diesel hikes (`N003`), and trap notes (`N006`/`N010`).
- **Check 4 (3-Run Reproducibility):** 100% identical numbers, flags, verdicts, and matched note IDs across all 3 untouched runs.

---

## 6. Token Usage & Cost Report

By filtering 728 route-weeks down to 24 anomalies quantitatively and skipping LLM calls when 0 candidate notes match the route/date window, a full run requires only **16 LLM calls**:

| Metric | Value (Per Full Run) |
| :--- | :--- |
| **Open-Source Model** | `qwen/qwen3.8-27b` (via Groq API) |
| **Total Route-Weeks Evaluated** | 728 |
| **Anomalies Flagged for RAG** | 24 |
| **Total LLM API Calls** | 16 (8 fast-pathed with 0 candidate notes) |
| **Input (Prompt) Tokens** | 9,857 |
| **Output (Completion) Tokens** | 1,457 |
| **Total Tokens** | 11,314 |
| **Published Rate (Groq USD / 1M Tokens)** | $0.20 Input / $0.60 Output |
| **Total Cost per Full Run** | **$0.002846 USD (~0.28 cents)** |

---

## 7. Setup & How to Run

### 1. Install Dependencies

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### 2. Configure API Key
Create a `.env` file in the project root with your free Groq API key:

    GROQ_API_KEY=gsk_your_api_key_here

### 3. Run the Full Pipeline

    python main.py

Outputs `output/shipping_cost_anomalies.csv` and `output/token_cost_log.json`.

### 4. Run the Evaluation Suite & 3-Run Reproducibility Check

    python evaluate.py

### 5. Bonus Stretch Goal: Interactive Plain-English Q&A
Ask questions directly against the generated findings:

    python ask_assistant.py "Why did Chennai-Bangalore get pricier in February and March, and was it justified?"

Or launch interactive mode:

    python ask_assistant.py