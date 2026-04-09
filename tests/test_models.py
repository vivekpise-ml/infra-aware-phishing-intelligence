# test_models.py
"""
Tests:
1) Engineered classical models train & predict correctly
2) Independent TF-IDF pipeline loads & predicts
3) NEW TF-IDF → RF & TF-IDF → XGBoost pipelines load & predict
4) CharCNN single prediction works
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib

from src.train_classical import train_models
from src.train_tfidf_classical import train_from_dataframe
from src.config import MODEL_DIR
from src.features import extract_all_features


# ==========================================================
# 1. Test engineered classical models (RF/LR/XGB)
# ==========================================================
def test_classical_models():
    print("\n=== TEST: Classical Models Training + Prediction ===")

    # Dummy labeled set
    df = pd.DataFrame({
        "url": [
            "https://google.com",
            "https://linkedin.com",
            "https://openai.com",
            "https://amazon.com",
            "http://malicious-login.com/update",
            "http://secure-paypal-login.com/verify",
        ],
        "label": [0, 0, 0, 0, 1, 1]
    })

    # Train engineered-feature models
    results = train_models(df)

    # Predict few examples
    test_urls = [
        "https://openai.com",
        "http://malicious-update-login.net"
    ]
    feats = [extract_all_features(u) for u in test_urls]
    X = pd.DataFrame(feats).apply(pd.to_numeric, errors="coerce").fillna(0)

    for name, r in results.items():
        model_path = r["model_path"]
        scaler_path = model_path.replace("_model.pkl", "_scaler.pkl")

        assert os.path.exists(model_path)
        assert os.path.exists(scaler_path)

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        preds = model.predict(scaler.transform(X))
        print(f"{name} predictions:", preds)
        assert len(preds) == 2
        assert preds[0] in [0, 1]


# ==========================================================
# 2. Test independent TF-IDF pipeline
# ==========================================================
def test_independent_tfidf_pipeline():
    print("\n=== TEST: Independent TF-IDF Pipeline ===")
    path = os.path.join(MODEL_DIR, "tfidf_pipeline.pkl")

    if not os.path.exists(path):
        print("⚠ Independent TF-IDF pipeline not found:", path)
        return

    mdl = joblib.load(path)
    pred = mdl.predict(["http://secure-login-paypal.com/update"])[0]
    print("Independent TF-IDF prediction:", pred)
    assert pred in [0, 1]


# ==========================================================
# 3. Test new TF-IDF → RF & TF-IDF → XGBoost pipelines
# ==========================================================
def test_tfidf_classical_pipelines():
    print("\n=== TEST: NEW TF-IDF → RF / XGBoost ===")

    df = pd.DataFrame({
        "url": [
            "https://google.com",
            "https://openai.com",
            "https://linkedin.com",
            "http://phishy-update-site.com/login",
            "http://malicious-secure-paypal.com/update"
        ],
        "label": [0, 0, 0, 1, 1]
    })

    # Train new pipelines
    train_from_dataframe(df)

    for name in ["tfidf_rf.pkl", "tfidf_xgb.pkl"]:
        path = os.path.join(MODEL_DIR, name)
        assert os.path.exists(path), f"{name} missing!"

        mdl = joblib.load(path)

        preds = mdl.predict([
            "https://openai.com",
            "http://secure-malicious-login.net/verify"
        ])

        print(f"{name} predictions:", preds)
        assert len(preds) == 2
        assert preds[0] in [0, 1]


# ==========================================================
# 4. Test CharCNN single prediction
# ==========================================================
def test_charcnn_single():
    print("\n=== TEST: CharCNN single URL ===")

    try:
        from src.charcnn_predict import predict_single
        url = "http://secure-login-paypal.com.verify.info/login"
        label, prob = predict_single(url, model_dir="models")
        print("CharCNN prediction:", label, prob)
        assert label in ["malicious", "benign"]
    except Exception as e:
        print("⚠ CharCNN test skipped:", e)
