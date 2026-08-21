"""
generate_sample_data.py
------------------------
Creates a synthetic labeled URL dataset (data/urls.csv) so the full
pipeline can run end-to-end without downloading anything.

>>> REPLACE THIS FOR YOUR ACTUAL SUBMISSION <<<
For your real project, download a proper dataset instead, e.g.:
  - Kaggle: "Phishing Site URLs" (https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
  - UCI: "Phishing Websites Data Set"
Then place it at data/urls.csv with columns: url,label (label: 1=phishing, 0=legit)
"""

import random
import pandas as pd

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "wikipedia.org", "github.com", "amazon.com", "microsoft.com",
    "nytimes.com", "bbc.com", "stackoverflow.com", "linkedin.com", "apple.com",
    "reddit.com", "netflix.com", "spotify.com", "dropbox.com", "adobe.com",
]

LEGIT_PATHS = ["", "/about", "/products", "/blog/2024/update", "/help/faq",
               "/user/profile", "/search?q=example", "/docs/api", "/contact"]

PHISH_BRANDS = ["paypal", "apple", "bankofamerica", "netflix", "amazon",
                "microsoft", "chase", "wellsfargo", "instagram", "facebook"]

PHISH_TEMPLATES = [
    "http://{brand}-login-secure.com/verify/{rand}",
    "http://{brand}.com-account-update.info/signin",
    "http://192.168.{a}.{b}/{brand}/login.php",
    "http://secure-{brand}-verify.tk/confirm?id={rand}",
    "http://{brand}support.{tld}/webscr?cmd=login&{rand}=1",
    "http://bit.ly/{rand}",
    "http://{brand}-{rand}.xyz/update-password",
    "http://verify-{brand}-account.com/@{rand}/secure",
]

RAND_TLDS = ["tk", "ml", "ga", "cf", "top", "click"]


def rand_str(n=6):
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=n))


def make_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = "https"
    sub = random.choice(["", "www.", "www.", "docs."])  # www weighted common
    return f"{scheme}://{sub}{domain}{path}"


def make_phish_url():
    template = random.choice(PHISH_TEMPLATES)
    return template.format(
        brand=random.choice(PHISH_BRANDS),
        rand=rand_str(random.randint(4, 10)),
        a=random.randint(1, 255),
        b=random.randint(1, 255),
        tld=random.choice(RAND_TLDS),
    )


def generate(n_per_class=1500):
    urls, labels = [], []
    for _ in range(n_per_class):
        urls.append(make_legit_url())
        labels.append(0)
        urls.append(make_phish_url())
        labels.append(1)

    df = pd.DataFrame({"url": urls, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate(n_per_class=1500)
    df.to_csv("data/urls.csv", index=False)
    print(f"Saved {len(df)} rows to data/urls.csv")
    print(df["label"].value_counts())
