import pandas as pd

def main():
    results_path = "models/chronological_benchmark_results.csv"
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print("Benchmark results file not found. Please run model training first.")
        return

    print("="*60)
    print("PHASE 4: EVALUATION & SUMMARY REPORT")
    print("="*60)

    # 1. Best model per site (lowest RMSE)
    print("\n--- Best Model per Agro-Ecological Zone ---")
    best_per_site = df.loc[df.groupby("Site")["RMSE"].idxmin()]
    print(best_per_site[["Site", "Feature_Set", "Model", "RMSE", "MAE", "R2"]].to_string(index=False))

    # 2. Impact of Feature Sets across all sites
    print("\n--- Average Performance by Feature Set ---")
    feature_summary = df.groupby("Feature_Set")[["RMSE", "MAE", "R2"]].mean().reset_index()
    print(feature_summary.to_string(index=False))

    # 3. Overall best algorithm performance
    print("\n--- Average Performance by Algorithm ---")
    model_summary = df.groupby("Model")[["RMSE", "MAE", "R2"]].mean().reset_index()
    print(model_summary.to_string(index=False))

if __name__ == "__main__":
    main()