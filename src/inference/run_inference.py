# src/inference/run_inference.py

from urllib.parse import urlparse

from src.inference.predictor import PhishingPredictor
from src.inference.report_generator import ReportGenerator
from src.infrastructure.infra_features import get_infra_features
from src.drift.drift_detector import compute_drift_score

from src.explainability.shap_explainer import ShapExplainer


def run(url):
    predictor = PhishingPredictor()
    reporter = ReportGenerator()
    explainer = ShapExplainer()

    # Step 1: Prediction
    pred = predictor.predict(url)

    # Step 2: Extract domain properly
    domain = urlparse(url).netloc
    print("DEBUG DOMAIN:", domain) # debug

    # Step 3: Infra intelligence (uses domain, NOT full URL)
    infra = get_infra_features(domain)

    print("INFRA:", infra)

    # Step 4: Drift
    drift = compute_drift_score(pred["features"], None)

    # Step 5: Explanation (placeholder for now)
   
    explanation_list = explainer.explain(
        pred["X"],
        predictor.feature_names
    )

    explanation = "\n".join(explanation_list)

    # Step 6: Generate report
    report = reporter.generate(pred, infra, drift, explanation)

    # return report

    return {
        "report": report,
        "risk_score": pred.get("risk_score"),
        "label": pred.get("label"),
        "risk_tier": pred.get("risk_tier"),
        "explanation_list": explanation_list
    }


if __name__ == "__main__":
    url = "http://secure-paypal-login.com/update"
    report = run(url)
    print(report)