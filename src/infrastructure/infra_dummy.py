# src/infrastructure/infra_dummy.py
"""
Dummy infrastructure feature provider.

Purpose:
- Fast execution (no network / WHOIS calls)
- Used when USE_REAL_INFRA = False
- Keeps feature schema consistent with real infra_features.py
"""


def get_infra_features(domain: str) -> dict:
    """
    Return default (neutral) infrastructure features.

    Parameters:
        domain (str): extracted domain (not used in dummy)

    Returns:
        dict: numeric infra features
    """

    return {
        # Domain age in days (0 = unknown / neutral)
        "domain_age_days": 0,

        # Whether domain resolves to an IP
        "has_ip": 0,

        # Whether ASN information is available
        "has_asn": 0,

        # Whether hosting is foreign (0 = neutral)
        "is_foreign_hosting": 0,
    }