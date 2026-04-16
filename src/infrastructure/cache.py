import json
import os
import time

CACHE_PATH = "cache/infra_cache.json"
TTL_SECONDS = 60 * 60 * 24  # 24 hours

def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r") as f:
        return json.load(f)

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

def get_cached(domain):
    cache = load_cache()
    entry = cache.get(domain)

    if not entry:
        return None

    if time.time() - entry["timestamp"] > TTL_SECONDS:
        return None

    return entry["data"]

def set_cache(domain, data):
    cache = load_cache()
    cache[domain] = {
        "data": data,
        "timestamp": time.time()
    }
    save_cache(cache)