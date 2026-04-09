# src/inference/predictor.py

import joblib
import pandas as pd
from src.features import extract_all_features
from src.charcnn_predict import predict_single
from src.config import MODEL_DIR

class PhishingPredictor:

    def __init__(self):
        # Load main model (XGBoost)
        self.model = joblib.load(f"{MODEL_DIR}/xgboost_model.pkl")
        self.scaler = joblib.load(f"{MODEL_DIR}/classical_scaler.pkl")
        self.feature_names = joblib.load(f"{MODEL_DIR}/classical_feature_names.pkl")

    def extract_features(self, url, html=None):
        feats = extract_all_features(url, html)
        X = pd.DataFrame([feats])

        # Align features
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_names]
        X = self.scaler.transform(X)

        return X, feats

    def predict(self, url, html=None):
        X, raw_features = self.extract_features(url, html)

        prob = self.model.predict_proba(X)[0][1]
        label = "malicious" if prob > 0.5 else "benign"

        risk_tier = PhishingPredictor.get_risk_tier(prob)

        return {
            "url": url,
            "label": label,
            "risk_score": float(prob),
            "features": raw_features,
            "risk_tier": risk_tier,
            "raw_features": raw_features,
            "X" : X
        }
    
    @staticmethod
    def get_risk_tier(score):
        if score > 0.8:
            return "HIGH"
        elif score > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
        
