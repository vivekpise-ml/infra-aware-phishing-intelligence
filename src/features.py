# features.py
"""
Feature extraction module for phishing detection.
Includes:
- URL-based features (length, entropy, keyword checks, structure, lexical statistics, etc.)
- HTML-based features (forms, scripts, iframes, event handlers)
- Combines both into a single feature dictionary
"""

import re
import math
from collections import Counter
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import tldextract

# -------------------------------------------------------------------
# Known URL shorteners
# -------------------------------------------------------------------
SHORTENER_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "buff.ly", "is.gd", "rebrand.ly", "lnkd.in", "shorturl.at"
]

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy (measure of randomness) of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c / len(s) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def is_shortener(url: str) -> int:
    """Check if URL belongs to known shortening services."""
    return int(any(dom in url.lower() for dom in SHORTENER_DOMAINS))

def has_ip_address(domain: str) -> int:
    """Check if the domain is an IP address."""
    return int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)))

def count_special_chars(s):
    specials = r"!@#$%^&*()_+\-={}[\]|\\:;\"'<>,.?/~`"
    return sum([1 for c in s if c in specials])

def count_encoded_chars(s):
    return s.count('%')

# -------------------------------------------------------------------
# ⭐ Enhanced URL feature extraction (70+ lexical features)
# -------------------------------------------------------------------
def extract_url_features(url: str) -> dict:
    """
    Extract rich numerical and lexical features from a URL.
    This is an upgraded and expanded version of the earlier extractor,
    producing ~70 features for high ML accuracy.
    """

    features = {}

    parsed = urlparse(url)
    full = url
    netloc = parsed.netloc
    path = parsed.path
    query = parsed.query

    # ---------------------------
    # Domain info via tldextract
    # ---------------------------
    tld_info = tldextract.extract(url)
    domain = f"{tld_info.domain}.{tld_info.suffix}" if tld_info.suffix else tld_info.domain
    #features["domain"] = domain
    features["domain"] = len(domain)  #making it numerical instead of string
    features["is_shortener"] = is_shortener(domain)

    # ---------------------------
    # Basic lengths
    # ---------------------------
    features["url_length"] = len(full)
    features["domain_length"] = len(netloc)
    features["path_length"] = len(path)
    features["query_length"] = len(query)

    # ---------------------------
    # Character counts
    # ---------------------------
    features["dot_count"] = full.count(".")
    features["slash_count"] = full.count("/")
    features["dash_count"] = full.count("-")
    features["underscore_count"] = full.count("_")
    features["at_count"] = full.count("@")
    features["question_count"] = full.count("?")
    features["equal_count"] = full.count("=")
    features["digit_count"] = sum(c.isdigit() for c in full)
    features["letter_count"] = sum(c.isalpha() for c in full)
    features["encoded_count"] = count_encoded_chars(full)
    features["special_count"] = count_special_chars(full)

    # ---------------------------
    # Ratios
    # ---------------------------
    url_len = features["url_length"]
    features["digit_ratio"] = features["digit_count"] / url_len if url_len else 0
    features["letter_ratio"] = features["letter_count"] / url_len if url_len else 0
    features["special_ratio"] = features["special_count"] / url_len if url_len else 0
    features["encoded_ratio"] = features["encoded_count"] / url_len if url_len else 0

    # ---------------------------
    # Entropy
    # ---------------------------
    features["url_entropy"] = shannon_entropy(full)
    features["domain_entropy"] = shannon_entropy(netloc)
    features["path_entropy"] = shannon_entropy(path)

    # ---------------------------
    # Domain structure
    # ---------------------------
    features["subdomain_count"] = len(netloc.split(".")) - 2 if "." in netloc else 0
    features["has_ip_domain"] = has_ip_address(netloc)
    features["tld_length"] = len(tld_info.suffix) if tld_info.suffix else 0

    # ---------------------------
    # Path/query structure
    # ---------------------------
    features["path_depth"] = path.count("/")
    features["param_count"] = query.count("&") + 1 if query else 0
    features["key_value_pairs"] = query.count("=")

    # ---------------------------
    # Token-based features
    # ---------------------------
    tokens = re.split(r"[./?=&_-]", full)
    token_lengths = [len(t) for t in tokens if t]

    features["num_tokens"] = len(token_lengths)
    features["avg_token_len"] = sum(token_lengths)/len(token_lengths) if token_lengths else 0
    features["max_token_len"] = max(token_lengths) if token_lengths else 0

    # ---------------------------
    # Keyword indicators
    # ---------------------------
    keywords = [
        "secure", "account", "update", "login", "signin", "bank",
        "verify", "free", "bonus", "gift", "click", "download",
        "paypal", "confirm", "password", "ebay", "amazon",
        "support", "alert", "billing", "invoice"
    ]
    for kw in keywords:
        features[f"kw_{kw}"] = int(kw in full.lower())

    # ---------------------------
    # Suspicious patterns
    # ---------------------------
    features["has_php"] = int(".php" in full)
    features["has_html"] = int(".html" in full)
    features["has_https_in_domain"] = int("https" in netloc)
    features["double_slash"] = int("//" in path)
    features["dot_in_path"] = int("." in path)
    features["long_subdomain"] = int(features["subdomain_count"] >= 3)

    return features

# -------------------------------------------------------------------
# HTML feature extraction (same as original)
# -------------------------------------------------------------------
def extract_html_features(html: str) -> dict:
    """Extract features from HTML content."""
    soup = BeautifulSoup(html or "<html></html>", "html.parser")
    features = {}

    features["num_forms"] = len(soup.find_all("form"))
    features["num_iframes"] = len(soup.find_all("iframe"))
    features["num_scripts"] = len(soup.find_all("script"))
    features["num_links"] = len(soup.find_all("a"))
    features["has_event_handlers"] = int(
        bool(soup.find_all(attrs={"onload": True})) or bool(soup.find_all(attrs={"onclick": True}))
    )

    return features

# -------------------------------------------------------------------
# Combined feature extraction (still the same interface)
# -------------------------------------------------------------------
def extract_all_features(url: str, html: str = None) -> dict:
    """
    Combine both URL and HTML-based features into a single dictionary.
    If HTML is missing, HTML features are filled with zeros.
    """
    features = extract_url_features(url)

    if html:
        features.update(extract_html_features(html))
    else:
        # Fill missing HTML features with zeros
        features.update({
            "num_forms": 0,
            "num_iframes": 0,
            "num_scripts": 0,
            "num_links": 0,
            "has_event_handlers": 0,
        })

    return features

# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample_url = "https://tinyurl.com/2p9fb9kw"
    sample_html = "<html><body><form></form><script></script></body></html>"

    feats = extract_all_features(sample_url, sample_html)
    for k, v in feats.items():
        print(f"{k}: {v}")
