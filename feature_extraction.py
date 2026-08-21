"""
feature_extraction.py
----------------------
Extracts lexical / URL-based features for phishing detection.
No live network calls (no WHOIS, no page scraping) so it's fast
and works offline - suitable for a 2-week internship timeline.
"""

import re
import pandas as pd
from urllib.parse import urlparse

# Common URL shortener domains
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "cutt.ly", "rb.gy"
}

# Words commonly abused in phishing URLs
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank",
    "confirm", "signin", "webscr", "ebayisapi", "password", "pay"
]

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


def extract_features(url: str) -> dict:
    """Extract a dict of numeric/binary features from a single URL."""
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    features = {}

    # --- Length-based features ---
    features["url_length"] = len(url)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)

    # --- Character counts ---
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_at"] = url.count("@")
    features["num_question_marks"] = url.count("?")
    features["num_equal_signs"] = url.count("=")
    features["num_underscores"] = url.count("_")
    features["num_ampersands"] = url.count("&")
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_slashes"] = url.count("/")
    features["num_subdomains"] = max(hostname.count(".") - 1, 0)

    # --- Binary / boolean features ---
    features["has_ip_address"] = int(bool(IP_PATTERN.match(hostname)))
    features["has_https"] = int(parsed.scheme == "https")
    features["has_port"] = int(parsed.port is not None)
    features["is_shortened"] = int(hostname in SHORTENERS)
    features["has_suspicious_word"] = int(
        any(word in url.lower() for word in SUSPICIOUS_WORDS)
    )
    features["double_slash_redirect"] = int(url.rfind("//") > 7)
    features["has_at_symbol"] = int("@" in url)
    features["digit_letter_ratio"] = (
        features["num_digits"] / len(url) if len(url) > 0 else 0
    )

    return features


def build_feature_dataframe(urls: list, labels: list = None) -> pd.DataFrame:
    """Build a full feature DataFrame from a list of URLs (+ optional labels)."""
    rows = [extract_features(u) for u in urls]
    df = pd.DataFrame(rows)
    if labels is not None:
        df["label"] = labels
    return df


if __name__ == "__main__":
    # quick smoke test
    sample_urls = [
        "http://192.168.1.1/login/verify-account",
        "https://www.google.com",
        "http://bit.ly/3xYzAbc",
        "https://secure-paypal-login.com-update.info/signin",
    ]
    df = build_feature_dataframe(sample_urls)
    print(df.to_string())
