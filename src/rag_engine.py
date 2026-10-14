import json
import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

_CACHED_MODEL = None


def get_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please add GROQ_API_KEY=gsk_... to your .env file."
        )
    return api_key.strip()


def get_active_model(api_key):
    """Automatically fetch available models from Groq and pick the best active text model."""
    global _CACHED_MODEL
    if _CACHED_MODEL:
        return _CACHED_MODEL

    env_model = os.getenv("GROQ_MODEL")
    if env_model:
        _CACHED_MODEL = env_model.strip()
        return _CACHED_MODEL

    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(GROQ_MODELS_URL, headers=headers, timeout=10)
    resp.raise_for_status()
    models = [m["id"] for m in resp.json().get("data", [])]

    # Filter out audio/TTS/guard models
    skip_keywords = ("whisper", "guard", "tts", "audio", "vision", "embed")
    text_models = [
        m for m in models
        if not any(k in m.lower() for k in skip_keywords)
    ]

    # Preferred order of open-source models
    preferred_keywords = ["llama-3", "llama3", "llama-4", "llama", "qwen", "gemma", "mixtral", "deepseek"]
    for kw in preferred_keywords:
        for m in text_models:
            if kw in m.lower():
                _CACHED_MODEL = m
                return _CACHED_MODEL

    if text_models:
        _CACHED_MODEL = text_models[0]
        return _CACHED_MODEL

    raise RuntimeError(f"No suitable text model found in Groq account. Available: {models}")


def load_notes(filepath):
    """Load and validate contextual notes."""
    df = pd.read_csv(filepath)
    required_columns = {"note_id", "date", "applies_to", "note"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in context notes: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["applies_to"] = df["applies_to"].fillna("").astype(str).str.strip()
    df["note_id"] = df["note_id"].fillna("").astype(str).str.strip()
    df["note"] = df["note"].fillna("").astype(str).str.strip()
    return df

# it helps to find notes for a route so that if the note is not present then there is no need to send the data to AI and waste tokens
def find_candidate_notes(route, week_of, notes_df):
    """
    Find notes relevant to a route within a reasonable window around the target week.
    Includes a 28-day lookback and 7-day lookahead within that week.
    """
    if " to " in route:
        origin, dest = route.split(" to ", 1)
    elif "-" in route:
        origin, dest = route.split("-", 1)
    else:
        raise ValueError("Route must use 'Origin to Destination' or 'Origin-Destination' format.")

    route_dash = f"{origin.strip()}-{dest.strip()}"

    relevant = notes_df[
        notes_df["applies_to"].str.casefold().isin(
            [route_dash.casefold(), "all routes"]
        )
    ]

    week_start = pd.to_datetime(week_of)
    window_start = week_start - pd.Timedelta(days=28)
    window_end = week_start + pd.Timedelta(days=7)

    return relevant[
        relevant["date"].between(window_start, window_end, inclusive="both")
    ].copy()


def generate_explanation(
    api_key,
    route,
    week_of,
    current_cost,
    historical_cost,
    peer_cost,
    candidate_notes_df,
    model_name=None,
):
    """
    Generate a grounded RAG explanation using Groq.

    Returns:
        (verdict, matched_note_id, reason, token_usage_dict)
    """
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0}

    if candidate_notes_df.empty:
        return (
            "Yes",
            "",
            "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review.",
            empty_usage,
        )

    if not model_name:
        model_name = get_active_model(api_key)

    candidate_ids = []
    note_blocks = []

    for _, row in candidate_notes_df.iterrows():
        note_id = str(row["note_id"]).strip()
        if not note_id:
            continue
        if note_id not in candidate_ids:
            candidate_ids.append(note_id)

        note_date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        note_blocks.append(
            f"Note ID: {note_id} | Date: {note_date} | Applies to: {row['applies_to']}\n"
            f"Content: {row['note']}"
        )

    notes_text = "\n\n".join(note_blocks)
    hist_str = f"{historical_cost:.2f}" if pd.notna(historical_cost) else "N/A"
    peer_str = f"{peer_cost:.2f}" if pd.notna(peer_cost) else "N/A"

    system_prompt = (
        "You are a strict freight cost auditing assistant. You evaluate whether a shipping cost "
        "increase is justified by real-world context notes. You never hallucinate or guess. "
        "You always respond in valid JSON format."
    )

    user_prompt = f"""Evaluate this shipping route cost spike against the retrieved context notes.

TARGET ANOMALY:
- Route: {route}
- Week of (Monday): {week_of} (covers Monday {week_of} through Sunday of that week)
- Current Cost per tonne-km: {current_cost:.2f} ({ current_cost / historical_cost - 1:+.1%} vs history)
- 8-Week Historical Baseline: {hist_str}
- Peer Route Baseline: {peer_str}

CANDIDATE CONTEXT NOTES:
{notes_text}

STRICT AUDIT RULES:
1. A cost spike is ONLY justified ("No (justified)") if a note describes a genuine cost-increasing event (e.g., flooding/detours, festival surcharge, nationwide diesel price hike) that applies to this route AND was actively in effect during the week starting {week_of}.
2. CAREFULLY CHECK DATES INSIDE NOTE TEXT: If a note says a disruption ended BEFORE {week_of} (for example, flooding from Feb 24 to Mar 8 where roads were passable by Mar 9 does NOT justify spikes on 2025-03-10 or 2025-03-17), you MUST return verdict "Yes" and note_id "".
3. If a note says demand was stable, normal, roads improved, or costs were absorbed/unaffected, it does NOT justify a cost rise. Return verdict "Yes", note_id "", and briefly mention why that closest note (citing its ID and date) does not justify the increase.
4. Allowed note IDs when justified: {json.dumps(candidate_ids)}. When verdict is "Yes", note_id MUST be "".

Return ONLY a JSON object with these exact keys:
{{
  "verdict": "No (justified)" or "Yes",
  "note_id": "exact matched Note ID if justified, otherwise empty string",
  "reason": "1-2 concise plain-English sentences. If justified, state 'Matches note <ID> dated <date>: ...'. If unexplained ('Yes'), explain why no note (or the closest candidate note <ID>) justifies the hike and flag for review."
}}"""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "seed": 42,
        "response_format": {"type": "json_object"},
        "max_tokens": 256,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                GROQ_API_URL, headers=headers, json=payload, timeout=20
            )
        except requests.exceptions.RequestException as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Groq connection failed after {MAX_RETRIES} attempts: {exc}")

        if response.status_code in RETRYABLE_STATUS_CODES:
            wait_time = 2 ** (attempt + 1)
            print(f"Groq rate limit/busy (HTTP {response.status_code}). Waiting {wait_time}s...")
            time.sleep(wait_time)
            continue

        if not response.ok:
            raise RuntimeError(f"Groq API Error {response.status_code}: {response.text}")

        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"].strip()
        raw_usage = res_json.get("usage", {})
        usage = {
            "prompt_tokens": raw_usage.get("prompt_tokens", 0),
            "completion_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
            "llm_calls": 1,
        }

        parsed = json.loads(content)
        raw_verdict = str(parsed.get("verdict", "")).strip()
        raw_note_id = str(parsed.get("note_id", "")).strip()
        reason = str(parsed.get("reason", "")).strip()

        valid_id_map = {cid.casefold(): cid for cid in candidate_ids}
        matched_note_id = valid_id_map.get(raw_note_id.casefold(), "")

        if raw_verdict.casefold() in {"no (justified)", "justified"} and matched_note_id:
            return "No (justified)", matched_note_id, reason, usage
        else:
            return "Yes", "", reason, usage

    raise RuntimeError("Groq API failed after maximum retries.")


if __name__ == "__main__":
    key = get_api_key()
    active_model = get_active_model(key)
    print(f"Testing Groq Open-Source RAG Engine (Auto-selected model: {active_model})...")

    notes = load_notes("data/context_notes.csv")

    route = "Ahmedabad to Mumbai"
    week = "2025-01-20"

    candidates = find_candidate_notes(route, week, notes)
    print(f"Candidate notes found: {len(candidates)}")

    verdict, note_id, reason, usage = generate_explanation(
        key, route, week, 3.29, 2.54, 2.68, candidates, model_name=active_model
    )

    print("\n--- Output ---")
    print("Flagged (Verdict):", verdict)
    print("Matched Note ID:  ", note_id)
    print("Reason:           ", reason)
    print("Token Usage:      ", usage)