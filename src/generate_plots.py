import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    results_path = "models/chronological_benchmark_results.csv"
    if not os.path.exists(results_path):
        print("Results file not found. Please run model training first.")
        return

    df = pd.read_csv(results_path)

    # Set publication-style aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    os.makedirs("reports/figures", exist_ok=True)

    # -------------------------------------------------------------
    # Figure 1: R2 Comparison by Feature Set and Site (CatBoost Focus)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    catboost_df = df[df["Model"] == "CatBoost"]
    
    ax = sns.barplot(
        data=catboost_df, 
        x="Site", 
        y="R2", 
        hue="Feature_Set", 
        palette="Blues_d"
    )
    plt.title("CatBoost $ET_0$ Estimation Performance ($R^2$) Across Nigerian Agro-Ecological Zones")
    plt.xlabel("Agro-Ecological Site")
    plt.ylabel("Coefficient of Determination ($R^2$)")
    plt.ylim(0.4, 1.05)
    plt.legend(title="Feature Set", loc="lower right")
    plt.tight_layout()
    
    fig1_path = "reports/figures/catboost_r2_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Overall Algorithm Performance Comparison (RMSE)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=df, 
        x="Feature_Set", 
        y="RMSE", 
        hue="Model", 
        palette="muted",
        errorbar=None
    )
    plt.title("Overall Error (RMSE) Across Feature Sets and Machine Learning Algorithms")
    plt.xlabel("Feature Input Combination")
    plt.ylabel("Root Mean Square Error (mm/day)")
    plt.legend(title="Algorithm", loc="upper right")
    plt.tight_layout()

    fig2_path = "reports/figures/algorithm_rmse_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig2_path}")

    print("\nAll benchmark visualization figures successfully generated and saved in reports/figures/!")

if __name__ == "__main__":
    main()