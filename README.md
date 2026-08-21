# Ai-Based-Url-detection-system-
AI-Based Phishing URL Detection System
A 2-week internship project: detects phishing URLs using lexical/URL-based features and a machine learning classifier, served through a simple web app.
Project Structure
phishing_detector/
├── data/
│   └── urls.csv              # dataset (url, label) — 1=phishing, 0=legit
├── models/
│   └── phishing_model.pkl    # trained model (created after training)
├── templates/
│   └── index.html            # web UI
├── feature_extraction.py     # extracts features from raw URLs
├── generate_sample_data.py   # creates a demo dataset (replace with real data)
├── train_model.py            # trains + evaluates Logistic Regression & Random Forest
├── app.py                    # Flask app: paste a URL, get a prediction
└── requirements.txt
Setup
pip install -r requirements.txt
⚠️ IMPORTANT: Replace the demo dataset before submitting
generate_sample_data.py creates a synthetic dataset so the pipeline runs end-to-end immediately. It's rule-generated, so the model scores 100% on it — that's not a real result and reviewers will notice.
For your real submission, download an actual dataset:
Kaggle: "Phishing Site URLs" — https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
UCI: "Phishing Websites Data Set" — https://archive.ics.uci.edu/dataset/327/phishing+websites
Save it as data/urls.csv with exactly two columns: url,label (label = 1 for phishing, 0 for legitimate), then re-run training below. With a real dataset, expect realistic accuracy in the 90-97% range — report that, and discuss it in your write-up (that's the actual point of the project).
How to run
(First time / demo only) Generate the sample dataset:
python3 generate_sample_data.py
Train the model:
python3 train_model.py
This prints accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and feature importances for both models, then saves the better one to models/phishing_model.pkl.
Run the web app:
python3 app.py
Open http://localhost:5000, paste a URL, click "Check URL".
Features used (all computed instantly, no network calls)
URL/hostname/path length
Counts of dots, hyphens, @, ?, =, _, &, digits, slashes
Number of subdomains
IP address used instead of domain name
HTTPS presence
Non-standard port
Known URL-shortener domains
Suspicious keywords (login, verify, secure, account, etc.)
Digit-to-length ratio
For your report
Problem statement: phishing URLs trick users into giving up credentials; blocklists alone can't catch new/unseen phishing sites.
Approach: extract lexical features from the URL string itself (no need to visit the page), train a classifier to generalize to unseen URLs.
Models compared: Logistic Regression (interpretable baseline) vs Random Forest (captures non-linear feature interactions).
Metrics: recall is prioritized — a missed phishing URL (false negative) is worse than a false alarm on a legit site.
Limitations: lexical features alone can be fooled by cleverly crafted legitimate-looking URLs; content-based features (page structure, forms, favicon mismatch) and domain-age lookups would improve robustness but need live network access.
Future work: add WHOIS domain-age features, page-content analysis, try a character-level LSTM/CNN, browser extension deployment.
Tech stack
Python, pandas, scikit-learn, Flask, HTML/CSS/JS