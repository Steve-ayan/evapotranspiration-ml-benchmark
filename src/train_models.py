import os
import glob
import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from src.metrics import evaluate_predictions

def main():
    processed_files = glob.glob("data/processed/*_processed.csv")
    if not processed_files:
        print("No processed files found in data/processed/. Please run data processing first.")
        return

    # Updated feature sets including engineering features like Delta_T and DOY
    feature_sets = {
        "Temp_Only": ["T2M_MAX", "T2M_MIN", "Delta_T"],
        "Temp_Rad": ["T2M_MAX", "T2M_MIN", "Delta_T", "ALLSKY_SFC_SW_DWN", "DOY"],
        "Full_Sensors": ["T2M_MAX", "T2M_MIN", "Delta_T", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M", "DOY"]
    }

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "CatBoost": CatBoostRegressor(iterations=100, random_seed=42, verbose=0)
    }

    all_results = []

    for file_path in processed_files:
        site_name = os.path.basename(file_path).split('_')[0]
        print(f"\n--- Processing Site (Chronological Split): {site_name} ---")
        df = pd.read_csv(file_path)

        # Feature Engineering: Delta T and Day of Year (DOY)
        if "T2M_MAX" in df.columns and "T2M_MIN" in df.columns:
            df["Delta_T"] = df["T2M_MAX"] - df["T2M_MIN"]
        
        # Ensure year or date column exists for chronological split
        if "YEAR" not in df.columns and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"])
            df["YEAR"] = df["DATE"].dt.year
            df["DOY"] = df["DATE"].dt.dayofyear
        elif "YEAR" not in df.columns:
            df["YEAR"] = range(1990, 1990 + len(df))

        target_col = "ET0_PM"
        if target_col not in df.columns:
            print(f"Target column {target_col} missing in {file_path}. Skipping.")
            continue

        # Chronological Split: 1990-2016 Train, 2017-2023 Test
        train_df = df[df["YEAR"] <= 2016]
        test_df = df[df["YEAR"] >= 2017]

        if train_df.empty or test_df.empty:
            print(f"Warning: Chronological split resulted in empty sets for {site_name}. Check year column values.")
            continue

        for fs_name, features in feature_sets.items():
            missing_feats = [f for f in features if f not in df.columns]
            if missing_feats:
                print(f"Skipping feature set {fs_name}: missing {missing_feats}")
                continue

            X_train = train_df[features]
            y_train = train_df[target_col]
            X_test = test_df[features]
            y_test = test_df[target_col]

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metrics = evaluate_predictions(y_test, preds)

                result_entry = {
                    "Site": site_name,
                    "Feature_Set": fs_name,
                    "Model": model_name,
                    **metrics
                }
                all_results.append(result_entry)
                print(f"[{fs_name}] {model_name} -> RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")

    results_df = pd.DataFrame(all_results)
    os.makedirs("models", exist_ok=True)
    results_df.to_csv("models/chronological_benchmark_results.csv", index=False)
    print("\nChronological benchmark complete! Results saved to models/chronological_benchmark_results.csv")

if __name__ == "__main__":
    main()