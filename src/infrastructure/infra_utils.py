import socket
import whois
import tldextract
import requests
from datetime import datetime
import json
import os

CACHE_PATH = "data/cache/infra_cache.json"


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def extract_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"


def get_ip(domain):
    try:
        return socket.gethostbyname(domain)
    except:
        return None


def get_whois_info(domain):
    try:
        w = whois.whois(domain)

        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if creation_date:
            age_days = (datetime.now() - creation_date).days
        else:
            age_days = None

        registrar = w.registrar

        return age_days, registrar

    except:
        return None, None
    


def get_asn_info(ip):
    if not ip:
        return None, None

    try:
        url = f"http://ip-api.com/json/{ip}"
        res = requests.get(url, timeout=5).json()

        asn = res.get("as", None)
        country = res.get("country", None)

        return asn, country

    except:
        return None, None
    
    
def get_infra_features(domain):

    cache = load_cache()

    # ✅ If already cached
    if domain in cache:
        return cache[domain]

    # ❌ Otherwise fetch
    ip = get_ip(domain)
    age, registrar = get_whois_info(domain)
    asn, country = get_asn_info(ip)

    result = {
        "domain_age_days": age if age else 0,
        "has_registrar": int(registrar is not None),
        "has_asn": int(asn is not None),
        "is_foreign_hosting": int(country not in ["India", "US"] if country else 0)
    }

    # 💾 Save to cache
    cache[domain] = result
    save_cache(cache)

    return result    