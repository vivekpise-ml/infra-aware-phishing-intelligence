# src/infrastructure/infra_features.py

import socket
import whois
import requests
import json
import os
from datetime import datetime

CACHE_PATH = "data/cache/infra_cache.json"


# ---------------------------------------
# CACHE HANDLING
# ---------------------------------------
def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

# globally loading cache
CACHE = load_cache()

# ---------------------------------------
# DOMAIN → IP
# ---------------------------------------
def get_ip(domain):
    try:
        return socket.gethostbyname(domain)
    except:
        return None


# ---------------------------------------
# WHOIS → DOMAIN AGE
# ---------------------------------------
def get_domain_age_days(domain):
    try:
        w = whois.whois(domain)
        creation = w.creation_date

        if isinstance(creation, list):
            creation = creation[0]

        if isinstance(creation, datetime):
            return (datetime.now() - creation).days
    except:
        pass
    return 0


# ---------------------------------------
# IP → ASN + COUNTRY
# ---------------------------------------
def get_ip_info(ip):
    if not ip:
        return None, None

    try:
        res = requests.get(f"http://ip-api.com/json/{ip}", timeout=5).json()
        asn = res.get("as", None)
        country = res.get("country", None)
        return asn, country
    except:
        return None, None


# ---------------------------------------
# MAIN FUNCTION
# ---------------------------------------
def get_infra_features(domain):

    #cache = load_cache()

    global CACHE

    domain = domain.replace("www.", "")

    # ✅ CACHE HIT
    if domain in CACHE:
        print("CACHE HIT:", domain)
        return CACHE[domain]

    # ❌ Otherwise fetch
    print("FETCHING:", domain)
    ip = get_ip(domain)
    #age, registrar = get_whois_info(domain)
    age = get_domain_age_days(domain)
    registrar = None # This is so because registrar is very unreliable. age is much better as a signal
    age = age if isinstance(age, int) else 0
    asn, country = get_ip_info(ip)

    features = {
        "domain_age_days": age if age else 0,
        "has_registrar": 0, # we removed it as get_whois_info() registrar is very unreliable
        "has_ip": int(ip is not None),
        "has_asn": int(asn is not None),
        "is_foreign_hosting": int(country not in ["India", "US"] if country else 0)
    }

    # 💾 SAVE to cache
    CACHE[domain] = features

    if len(CACHE) % 20 == 0:
        save_cache(CACHE)
    
    return features