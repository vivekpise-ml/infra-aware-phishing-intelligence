from datetime import datetime
from urllib.parse import urlparse

class ReportGenerator:

    def generate(self, prediction, infra=None, drift=None, explanation=None):

        infra = infra or {}
        drift = drift or {}

        url = prediction["url"]
        domain = urlparse(url).netloc

        risk = prediction["risk_score"]
        label = prediction["label"]
        tier = prediction.get("risk_tier", "UNKNOWN")

        report = f"""
════════════════════════════════════════
Phishing Risk Intelligence Report
════════════════════════════════════════

[1] DOMAIN SUMMARY
────────────────────────────────────────
Domain              : {domain}
Prediction          : {label.upper()}
Risk Score          : {risk:.2f}
Risk Tier           : {tier}

────────────────────────────────────────
[2] INFRASTRUCTURE INTELLIGENCE
────────────────────────────────────────
ASN                 : {infra.get('asn', 'N/A')}
Hosting Country     : {infra.get('country', 'N/A')}
Domain Age          : {infra.get('domain_age', 'N/A')}
Registrar           : {infra.get('registrar', 'N/A')}

────────────────────────────────────────
[3] DRIFT ANALYSIS
────────────────────────────────────────
Drift Score         : {drift.get('score', 'N/A')}
Drift Status        : {drift.get('status', 'N/A')}

────────────────────────────────────────
[4] MODEL EXPLANATION
────────────────────────────────────────
Top Factors:
{explanation}

────────────────────────────────────────
Model Version       : v1.0
Generated At        : {datetime.now()}
════════════════════════════════════════
"""
        return report