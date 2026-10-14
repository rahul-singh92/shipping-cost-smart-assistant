import pandas as pd


def flag_anomalies(
    weekly_df,
    hist_spike_pct=7.0,
    creep_hist_pct=5.0,
    creep_peer_pct=20.0,
):
    """
    Flags routes where shipping cost is rising in an out-of-the-ordinary way:
    1. Direct Historical Spike: >= 7.0% above the route's 8-week rolling average.
    2. Quiet Cost Creep: Actively rising (>= 5.0% vs own history) AND diverging
       significantly (>= 20.0%) above similar-length peer routes that week.
    """
    df = weekly_df.copy()

    # Calculate percentage differences vs baselines
    df["pct_vs_history"] = (
        (df["weekly_cost_per_tonne_km"] - df["historical_baseline"])
        / df["historical_baseline"]
        * 100.0
    )
    df["pct_vs_peers"] = (
        (df["weekly_cost_per_tonne_km"] - df["peer_baseline"])
        / df["peer_baseline"]
        * 100.0
    )

    # Rule 1: Sharp increase vs own trailing 8-week average
    cond_hist_spike = df["pct_vs_history"] >= hist_spike_pct

    # Rule 2: Gradual cost creep (rising vs own history AND far above peer routes)
    cond_peer_creep = (df["pct_vs_history"] >= creep_hist_pct) & (
        df["pct_vs_peers"] >= creep_peer_pct
    )

    df["is_anomaly"] = cond_hist_spike | cond_peer_creep

    # Format columns to match sample_output_format.csv contract
    df["route_formatted"] = df["route"].str.replace(" to ", "-", regex=False)
    df["week_of_str"] = pd.to_datetime(df["week_of"]).dt.strftime("%Y-%m-%d")
    df["cost_per_tonne_km_rounded"] = df["weekly_cost_per_tonne_km"].round(2)

    df["vs_own_history"] = df["pct_vs_history"].apply(
        lambda x: f"{x:+.1f}% vs this route's past average"
        if pd.notna(x)
        else "N/A (no prior history)"
    )
    df["vs_similar_routes"] = df["pct_vs_peers"].apply(
        lambda x: f"{x:+.1f}% vs similar-length routes this week"
        if pd.notna(x)
        else "N/A (no peer routes)"
    )

    flagged_df = df[df["is_anomaly"]].sort_values(by=["route_formatted", "week_of_str"]).copy()
    return flagged_df


if __name__ == "__main__":
    from src.data_pipeline import load_and_preprocess, calculate_baselines

    df_raw = load_and_preprocess("data/shipment_records.csv")
    df_baselines = calculate_baselines(df_raw)
    anomalies = flag_anomalies(df_baselines)

    print(f"Total flagged anomalies: {len(anomalies)}")
    print(
        anomalies[
            [
                "route_formatted",
                "week_of_str",
                "cost_per_tonne_km_rounded",
                "vs_own_history",
                "vs_similar_routes",
            ]
        ].to_string(index=False)
    )