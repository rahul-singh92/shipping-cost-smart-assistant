import pandas as pd
from main import run_pipeline

# Ground-truth labeled evaluation set covering all edge cases and guardrails
GROUND_TRUTH_CASES = [
    # (route, week_of, expected_flagged, expected_note_id, test_description)
    ("Delhi-Jaipur", "2024-11-11", "Yes", "", "Unexplained sudden spike (Sample row 1)"),
    ("Ahmedabad-Mumbai", "2025-01-20", "No (justified)", "N002", "Festival surcharge match (Sample row 2)"),
    ("Chennai-Bangalore", "2025-02-24", "No (justified)", "N001", "Active highway flooding week 1"),
    ("Chennai-Bangalore", "2025-03-03", "No (justified)", "N001", "Active highway flooding week 2"),
    ("Chennai-Bangalore", "2025-03-10", "Yes", "", "Post-flooding boundary check (roads passable Mar 9)"),
    ("Chennai-Bangalore", "2025-03-17", "Yes", "", "Trap note N009 (returned to normal conditions)"),
    ("Delhi-Jaipur", "2025-05-05", "No (justified)", "N003", "Nationwide diesel hike (+5-7%)"),
    ("Mumbai-Delhi", "2025-05-05", "No (justified)", "N003", "Nationwide diesel hike (+5-7%)"),
    ("Mumbai-Pune", "2025-09-15", "Yes", "", "Trap note N006 (stable demand - Sample row 3)"),
    ("Mumbai-Pune", "2025-10-27", "Yes", "", "Trap note N010 (compliance costs absorbed)"),
]

REQUIRED_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


def evaluate_single_run(df):
    """Validates schema contract, hallucination rules, and ground-truth accuracy."""
    print("\n=== RUNNING EVALUATION HARNESS ===")

    # 1. Schema Check
    cols_match = list(df.columns) == REQUIRED_COLUMNS
    print(f"[Check 1] Exact CSV column schema match: {'PASS' if cols_match else 'FAIL'}")
    assert cols_match, f"Expected {REQUIRED_COLUMNS}, got {list(df.columns)}"

    # 2. Hallucination / Contract Invariant Check
    # Every 'Yes' must have blank matched_note_id; every 'No (justified)' must have a valid note_id
    df_clean = df.fillna("")
    unexplained_with_note = df_clean[
        (df_clean["flagged"] == "Yes") & (df_clean["matched_note_id"] != "")
    ]
    justified_without_note = df_clean[
        (df_clean["flagged"] == "No (justified)") & (df_clean["matched_note_id"] == "")
    ]
    invariants_pass = len(unexplained_with_note) == 0 and len(justified_without_note) == 0
    print(f"[Check 2] Citation & verdict consistency invariants: {'PASS' if invariants_pass else 'FAIL'}")
    assert invariants_pass, "Found inconsistent flagged vs matched_note_id rows!"

    # 3. Ground-Truth Labeled Test Set Scoring
    passed = 0
    for route, week, exp_flag, exp_note, desc in GROUND_TRUTH_CASES:
        row = df_clean[(df_clean["route"] == route) & (df_clean["week_of"] == week)]
        if row.empty:
            print(f"  [FAIL] Missing expected anomaly row: {route} ({week}) - {desc}")
            continue
        actual_flag = row.iloc[0]["flagged"]
        actual_note = row.iloc[0]["matched_note_id"]
        if actual_flag == exp_flag and actual_note == exp_note:
            passed += 1
            print(f"  [PASS] {route} ({week}): {actual_flag} [{actual_note or 'blank'}] -> {desc}")
        else:
            print(
                f"  [FAIL] {route} ({week}): got ({actual_flag}, '{actual_note}'), "
                f"expected ({exp_flag}, '{exp_note}')"
            )

    print(f"\nGround-Truth Eval Score: {passed}/{len(GROUND_TRUTH_CASES)} ({passed/len(GROUND_TRUTH_CASES)*100:.1f}%)")
    assert passed == len(GROUND_TRUTH_CASES), "Ground truth evaluation did not achieve 100%!"


def run_reproducibility_check(num_runs=3):
    """Executes untouched runs and verifies identical numbers, flags, verdicts, and cited notes."""
    print(f"\n=== RUNNING {num_runs}-RUN REPRODUCIBILITY CHECK ===")
    run_dfs = []

    for i in range(1, num_runs + 1):
        out_csv = f"output/run_{i}_output.csv"
        out_log = f"output/run_{i}_cost_log.json"
        print(f"\n--- Executing Run {i}/{num_runs} ---")
        df, _ = run_pipeline(output_csv_path=out_csv, cost_log_path=out_log)
        run_dfs.append(df.fillna(""))

    # Also save Run 1 as the primary submission CSV
    run_dfs[0].to_csv("output/shipping_cost_anomalies.csv", index=False)

    # Compare deterministic columns across all 3 runs
    deterministic_cols = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "flagged",
        "matched_note_id",
    ]

    base_df = run_dfs[0][deterministic_cols]
    all_identical = True

    for i in range(1, num_runs):
        comp_df = run_dfs[i][deterministic_cols]
        if not base_df.equals(comp_df):
            all_identical = False
            diff = base_df.compare(comp_df)
            print(f"[FAIL] Diff detected between Run 1 and Run {i+1}:\n{diff}")
        else:
            print(f"[PASS] Run 1 vs Run {i+1}: 100% identical numbers, flags, verdicts, and matched_note_id.")

    if all_identical:
        print("\nSUCCESS: All 3 untouched runs produced 100% identical deterministic outputs!")
        evaluate_single_run(run_dfs[0])


if __name__ == "__main__":
    run_reproducibility_check(3)