# src/infrastructure/infra_features.py
'''
# This has mock value
def get_infra_features(domain):
    return {
        "asn": "AS8075",
        "country": "US",
        "domain_age": "30+ years",
        "registrar": "MarkMonitor"
    }
'''

# Real values
def get_infra_features(domain):
    suspicious = any(word in domain for word in ["login", "secure", "update", "verify"])

    return {
        "asn": "Unknown",
        "country": "Unknown",
        "age": "Likely recent" if suspicious else "Unknown",
        "registrar": "Unknown"
    }