import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.features_with_infra import extract_all_features
from src.config import DATA_PATH, MODEL_DIR


def train_xgboost_infra():

    df = pd.read_csv(DATA_PATH)

    # for infra feature
    # 🔥 ADD THIS RIGHT AFTER LOADING
    df = df.sample(3000, random_state=42)

    # detect label
    label_col = "label" if "label" in df.columns else "type"

    df[label_col] = df[label_col].astype(str).str.lower()
    df = df[df[label_col].isin(["benign", "malicious"])]

    y = df[label_col].map({"benign": 0, "malicious": 1})

    # extract features
    features = []
    for url in df["url"]:
        features.append(extract_all_features(url))

    X = pd.DataFrame(features).fillna(0)

    # save feature names
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(list(X.columns), f"{MODEL_DIR}/xgb_infra_feature_names.pkl")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, f"{MODEL_DIR}/xgb_infra_scaler.pkl")

    # model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    print(f"✅ XGBoost Accuracy: {acc:.4f}")

    joblib.dump(model, f"{MODEL_DIR}/xgboost_infra_model.pkl")
    print("💾 Saved XGBoost model")


if __name__ == "__main__":
    train_xgboost_infra()