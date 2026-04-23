import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.features import extract_all_features as extract_baseline
from src.features_with_infra import extract_all_features as extract_infra
from src.config import DATA_PATH


def prepare_features(urls, extractor):
    feats = []
    for url in urls:
        feats.append(extractor(url))
    return pd.DataFrame(feats).fillna(0)


def evaluate(model_path, X, y, model_name, scaler_path):

    model = joblib.load(model_path)

    scaler = joblib.load(scaler_path)

    # 🔥 SCALE FEATURES
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    print(f"\n🧠 {model_name}")
    print("Accuracy:", accuracy_score(y, preds))
    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    print("Classification Report:\n", classification_report(y, preds))


def main():

    df = pd.read_csv(DATA_PATH)

    label_col = "label" if "label" in df.columns else "type"
    df[label_col] = df[label_col].astype(str).str.lower()

    df = df[df[label_col].isin(["benign", "malicious"])]

    # 🔥 SAME DATA FOR BOTH
    df = df.sample(3000, random_state=42)

    y = df[label_col].map({"benign": 0, "malicious": 1})
    urls = df["url"].tolist()

    print("\n🔍 Extracting BASELINE features...")
    X_base = prepare_features(urls, extract_baseline)

    print("\n🔍 Extracting INFRA features...")
    X_infra = prepare_features(urls, extract_infra)

    # ⚠️ Align columns (IMPORTANT)
    base_cols = joblib.load("models/classical_feature_names.pkl")
    X_base = X_base.reindex(columns=base_cols, fill_value=0)

    infra_cols = joblib.load("models/xgb_infra_feature_names.pkl")
    X_infra = X_infra.reindex(columns=infra_cols, fill_value=0)

    #evaluate("models/xgboost_model.pkl", X_base, y, "Baseline Model")
    evaluate(
        "models/xgboost_model.pkl",
        X_base,
        y,
        "Baseline Model",
        "models/classical_scaler.pkl"
    )
    #evaluate("models/xgboost_infra_model.pkl", X_infra, y, "Infra Model")
    evaluate(
        "models/xgboost_infra_model.pkl",
        X_infra,
        y,
        "Infra Model",
        "models/xgb_infra_scaler.pkl"
    )


if __name__ == "__main__":
    main()