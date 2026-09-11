"""
Train Ridge regression and Random Forest to predict final allele frequency
from simulation parameters. Evaluate on held-out test set. Produce all figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "initial_allele_frequency",
    "temperature_rate",
    "selection_strength",
    "food_availability",
    "migration_rate",
    "population_size",
    "generations",
    "final_temperature_anomaly",
]

TARGET = "final_allele_frequency"


def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "evolution_dataset.csv"))
    # Drop rows where all replicates went extinct (target is NaN)
    df = df.dropna(subset=[TARGET])
    print(f"  Loaded {len(df)} rows with valid target")
    return df


def evaluate(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def main():
    # ------------------------------------------------------------------
    # Load & split
    # ------------------------------------------------------------------
    df = load_data()
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # Model 1: Ridge regression
    # ------------------------------------------------------------------
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    ridge_pipe.fit(X_train, y_train)
    ridge_pred = ridge_pipe.predict(X_test)
    ridge_metrics = evaluate(y_test, ridge_pred)
    print(f"  Ridge:  {ridge_metrics}")

    # ------------------------------------------------------------------
    # Model 2: Random Forest
    # ------------------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.85,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = evaluate(y_test, rf_pred)
    print(f"  RF:     {rf_metrics}")

    # ------------------------------------------------------------------
    # Select best model
    # ------------------------------------------------------------------
    if rf_metrics["MAE"] <= ridge_metrics["MAE"]:
        best_name = "RandomForest"
        best_pred = rf_pred
        best_metrics = rf_metrics
    else:
        best_name = "Ridge"
        best_pred = ridge_pred
        best_metrics = ridge_metrics

    print(f"  Best model: {best_name}")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    all_metrics = {
        "Ridge": ridge_metrics,
        "RandomForest": rf_metrics,
        "best_model": best_name,
    }
    with open(os.path.join(OUTPUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    comp_df = pd.DataFrame({
        "Model": ["Ridge", "RandomForest"],
        "MAE": [ridge_metrics["MAE"], rf_metrics["MAE"]],
        "RMSE": [ridge_metrics["RMSE"], rf_metrics["RMSE"]],
        "R2": [ridge_metrics["R2"], rf_metrics["R2"]],
    })
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("  Saved model_metrics.json and model_comparison.csv")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    # ------------------------------------------------------------------
    # Figure 3: Predicted vs Actual
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, best_pred, alpha=0.5, s=20, edgecolors="none")
    lims = [0, 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_xlabel("Simulated final allele frequency")
    ax.set_ylabel("Predicted final allele frequency")
    ax.set_title(f"Predicted vs Actual ({best_name})\nR² = {best_metrics['R2']}")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_actual.png"), dpi=150)
    plt.close(fig)
    print("  Saved predicted_vs_actual.png")

    # ------------------------------------------------------------------
    # Figure 4: Model comparison bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    models = ["Ridge", "Random Forest"]
    r2_vals = [ridge_metrics["R2"], rf_metrics["R2"]]
    bars = ax.bar(models, r2_vals, color=["#5B9BD5", "#70AD47"], width=0.5)
    ax.set_ylabel("R² (test set)")
    ax.set_title("Model Comparison: R²")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved model_comparison.png")

    # ------------------------------------------------------------------
    # Figure 5: Feature importance
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(fi_df["feature"], fi_df["importance"], color="#5B9BD5")
    ax.set_xlabel("Feature importance (Random Forest)")
    ax.set_title("Feature Importance for Predicting Final Allele Frequency")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("  Saved feature_importance.png")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== RESULTS SUMMARY ===")
    print(comp_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"\nTop features:")
    for _, row in fi_df.iloc[::-1].iterrows():
        print(f"  {row['feature']:30s}  {row['importance']:.4f}")


if __name__ == "__main__":
    main()
