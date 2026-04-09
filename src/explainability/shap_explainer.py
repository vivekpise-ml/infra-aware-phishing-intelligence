import shap
import joblib
import pandas as pd
from src.config import MODEL_DIR

class ShapExplainer:

    def __init__(self):
        self.model = joblib.load(f"{MODEL_DIR}/xgboost_model.pkl")
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, X, feature_names, top_k=5):
        shap_values = self.explainer.shap_values(X)

        vals = shap_values[0]
        pairs = list(zip(feature_names, vals))

        # Sort by absolute importance
        pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

        explanation = []

        for name, val in pairs[:top_k]:
            sign = "↑" if val > 0 else "↓"
            explanation.append(f"{sign} {name} ({val:.2f})")

        return explanation