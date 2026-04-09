import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------
# Load metrics JSON
# -------------------------------------------------------
def load_metrics(model_name, metrics_path="metrics"):
    file_path = f"{metrics_path}/{model_name}_metrics.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data).transpose()


# -------------------------------------------------------
# Visualization: report heatmap
# -------------------------------------------------------
def plot_classification_report(df, model_name, save_path="plots"):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 4))
    sns.heatmap(df.iloc[:-1, :-1], annot=True, cmap="Blues")
    plt.title(f"Classification Report Heatmap - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_report_heatmap.png", dpi=300)
    plt.close()


# -------------------------------------------------------
# Bar chart (precision/recall/F1)
# -------------------------------------------------------
def plot_metric_bars(df, model_name, save_path="plots"):
    os.makedirs(save_path, exist_ok=True)

    metrics_df = df.loc[['0', '1'], ['precision', 'recall', 'f1-score']]

    plt.figure(figsize=(8, 5))
    metrics_df.plot(kind='bar')
    plt.title(f"PRF Metrics - {model_name}")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_metric_bars.png", dpi=300)
    plt.close()


# -------------------------------------------------------
# Main function to generate all visualizations
# -------------------------------------------------------
def visualize_model(model_name):
    df = load_metrics(model_name)
    plot_classification_report(df, model_name)
    plot_metric_bars(df, model_name)
    print(f"Visualizations saved for {model_name}.")


if __name__ == "__main__":
    models = [
    "logisticregression",
    "randomforest",
    "xgboost",
    "tfidf_pipeline",
    "tfidf_rf",
    "tfidf_xgb",
    "charcnn"
    ]

    for m in models:
        visualize_model(m)
