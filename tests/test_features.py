# test_features.py
"""
Unit tests for feature extraction.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.features import extract_all_features

def test_url_features_basic():
    url = "https://tinyurl.com/abcd"
    feats = extract_all_features(url)
    assert "url_entropy" in feats
    assert "domain_entropy" in feats
    assert "path_entropy" in feats
    assert "is_shortener" in feats
    assert feats["is_shortener"] == 1

def test_html_features_counts():
    html = "<html><body><form></form><script></script><iframe></iframe></body></html>"
    feats = extract_all_features("https://example.com", html)
    assert feats["num_forms"] == 1
    assert feats["num_iframes"] == 1
    assert feats["num_scripts"] == 1
