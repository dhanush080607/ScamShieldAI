import pandas as pd
import re
import ipaddress

from urllib.parse import urlparse


# Load dataset
data = pd.read_csv("datasets/PhiUSIIL_Phishing_URL_Dataset.csv")


# Extract features from a URL
def extract_url_features(url):

    url = str(url).strip()

    # Handle URLs without http:// or https://
    if "://" not in url:
        url = "//" + url

    parsed = urlparse(url)
    domain = parsed.hostname or ""

    # Check whether domain is an IP address
    try:
        ipaddress.ip_address(domain)
        is_ip = 1
    except ValueError:
        is_ip = 0

    # Count URL characters
    url_length = len(url)
    digit_count = sum(char.isdigit() for char in url)
    letter_count = sum(char.isalpha() for char in url)
    special_count = len(re.findall(r"[^A-Za-z0-9]", url))

    # Extract domain information
    domain_parts = domain.split(".") if domain else []
    tld = domain_parts[-1] if domain_parts else ""

    subdomain_count = max(len(domain_parts) - 2, 0)

    # Suspicious words
    suspicious_words = [
        "login", "verify", "update", "secure",
        "account", "banking", "confirm",
        "password", "signin", "paypal"
    ]

    has_suspicious_word = int(
        any(word in url.lower() for word in suspicious_words)
    )

    return {
        "url_length": url_length,
        "domain_length": len(domain),
        "is_ip": is_ip,
        "tld_length": len(tld),
        "subdomain_count": subdomain_count,
        "is_https": int(parsed.scheme.lower() == "https"),
        "digit_count": digit_count,
        "letter_count": letter_count,
        "special_count": special_count,
        "at_count": url.count("@"),
        "hyphen_count": url.count("-"),
        "dot_count": url.count("."),
        "slash_count": url.count("/"),
        "question_count": url.count("?"),
        "equal_count": url.count("="),
        "percent_count": url.count("%"),
        "has_suspicious_word": has_suspicious_word
    }


# Extract features for every URL
features = data["URL"].apply(extract_url_features)

features_df = pd.DataFrame(features.tolist())

# Target labels
features_df["label"] = data["label"]

print("Extracted URL Features:")
print(features_df.head())

print("\nFeature Dataset Shape:")
print(features_df.shape)

print("\nMissing Values:")
print(features_df.isnull().sum().sum())

print("\nFeature Columns:")
print(features_df.columns.tolist())

# Save extracted features
features_df.to_csv(
    "datasets/url_features.csv",
    index=False
)

print("\nURL features saved successfully!")