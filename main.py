import json
import os
import time
import pandas as pd

from src.data_pipeline import load_and_preprocess, calculate_baselines
from src.anomaly_detector import flag_anomalies
from src.rag_engine import (
    get_api_key,
    get_active_model,
    load_notes,
    find_candidate_notes,
    generate_explanation,
)

# Published Groq API pricing per 1 Million tokens (USD) for cost reporting
MODEL_PRICING_PER_1M = {
    "default": {"input": 0.20, "output": 0.60},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "qwen/qwen3.8-27b": {"input": 0.20, "output": 0.60},
}


def run_pipeline(
    shipments_path="data/shipment_records.csv",
    notes_path="data/context_notes.csv",
    output_csv_path="output/shipping_cost_anomalies.csv",
    cost_log_path="output/token_cost_log.json",
):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print("1. Loading and preprocessing shipment records...")
    df_raw = load_and_preprocess(shipments_path)

    print("2. Computing weekly metrics and comparison baselines...")
    df_baselines = calculate_baselines(df_raw)

    print("3. Detecting cost anomalies...")
    anomalies_df = flag_anomalies(df_baselines)
    print(f"   Flagged {len(anomalies_df)} anomalous route-weeks out of {len(df_baselines)} total.")

    print("4. Initializing RAG Engine & loading context notes...")
    api_key = get_api_key()
    model_name = get_active_model(api_key)
    notes_df = load_notes(notes_path)
    print(f"   Using Open-Source Model: {model_name}")

    output_rows = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_llm_calls = 0

    for idx, row in anomalies_df.iterrows():
        route_raw = row["route"]
        route_fmt = row["route_formatted"]
        week_str = row["week_of_str"]
        curr_cost = row["weekly_cost_per_tonne_km"]
        hist_cost = row["historical_baseline"]
        peer_cost = row["peer_baseline"]

        candidates = find_candidate_notes(route_raw, week_str, notes_df)

        verdict, matched_note_id, reason, usage = generate_explanation(
            api_key=api_key,
            route=route_fmt,
            week_of=week_str,
            current_cost=curr_cost,
            historical_cost=hist_cost,
            peer_cost=peer_cost,
            candidate_notes_df=candidates,
            model_name=model_name,
        )

        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_llm_calls += usage["llm_calls"]

        output_rows.append(
            {
                "route": route_fmt,
                "week_of": week_str,
                "cost_per_tonne_km": f"{curr_cost:.2f}",
                "vs_own_history": row["vs_own_history"],
                "vs_similar_routes": row["vs_similar_routes"],
                "flagged": verdict,
                "matched_note_id": matched_note_id,
                "reason": reason,
            }
        )
        print(f"   [{len(output_rows)}/{len(anomalies_df)}] {route_fmt} ({week_str}) -> {verdict} [{matched_note_id}]")

    # Save output CSV matching sample_output_format.csv contract
    result_df = pd.DataFrame(output_rows)
    result_df.to_csv(output_csv_path, index=False)
    print(f"\n5. Saved final output to: {output_csv_path}")

    # Compute cost estimate
    pricing = MODEL_PRICING_PER_1M.get(model_name, MODEL_PRICING_PER_1M["default"])
    input_cost_usd = (total_prompt_tokens / 1_000_000) * pricing["input"]
    output_cost_usd = (total_completion_tokens / 1_000_000) * pricing["output"]
    total_cost_usd = input_cost_usd + output_cost_usd

    cost_summary = {
        "model": model_name,
        "total_route_weeks_evaluated": int(len(df_baselines)),
        "anomalies_flagged_for_rag": int(len(anomalies_df)),
        "llm_calls": total_llm_calls,
        "input_tokens": total_prompt_tokens,
        "output_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "published_rate_per_1m_tokens_usd": pricing,
        "estimated_total_cost_usd": round(total_cost_usd, 6),
    }

    with open(cost_log_path, "w") as f:
        json.dump(cost_summary, f, indent=2)

    print("\n--- Token & Cost Summary ---")
    print(json.dumps(cost_summary, indent=2))
    return result_df, cost_summary


if __name__ == "__main__":
    run_pipeline()