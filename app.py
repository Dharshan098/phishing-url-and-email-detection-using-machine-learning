# url_folder/app.py

import numpy as np
from flask import Blueprint, request, jsonify, send_from_directory
import pickle
import re
from urllib.parse import urlparse
from collections import Counter
import math
import os

# Create Blueprint
url_app = Blueprint(
    'url_app', __name__,
    template_folder='.',  # Not used here, but can be set for consistency
    static_folder='.'     # For serving static files like .css or .mp4
)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

# Feature extractor
def extract_features_from_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''

    features = []
    features.append(url.count('.'))
    features.append(url.count('/'))

    digits = sum(c.isdigit() for c in url)
    features.append(digits / len(url))

    special_chars = ['@', '=', '%', '?', '-', '_', '&']
    features.append(sum(url.count(c) for c in special_chars))

    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.cn']
    features.append(int(any(tld in url for tld in suspicious_tlds)))

    counter = Counter(url)
    probs = [freq / len(url) for freq in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    features.append(entropy)

    features.append(int(re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', hostname) is not None))

    keywords = ['login', 'secure', 'bank', 'update', 'account']
    features.append(int(any(k in url.lower() for k in keywords)))

    features.append(int(any(c * 4 in hostname for c in set(hostname))))

    features.append(int('//' in url.strip('/')[8:]))

    features.append(entropy * len(url))

    return features

# Routes

@url_app.route('/url')
def url_page():
    return send_from_directory(os.path.dirname(__file__), 'url.html')

@url_app.route('/url.css')
def serve_css():
    return send_from_directory(os.path.dirname(__file__), 'url.css')

@url_app.route('/playback1.mp4')
def serve_video():
    return send_from_directory(os.path.dirname(__file__), 'playback1.mp4', mimetype='video/mp4')

@url_app.route('/predict', methods=['POST'])
@url_app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url_input = data.get('url', '').strip().lower()

        if not url_input:
            return jsonify({'result': 'Please provide a URL'})

        # List of known phishing domain examples
        known_phishing_domains = {
            "secure-login-bankofamerica.com": "⚠️ This is a fake Bank of America login page.",
            "paypal-verification-alert.net": "⚠️ This domain is pretending to be PayPal for phishing.",
            "chase-online-update.info": "⚠️ This is a known phishing domain mimicking Chase bank.",
            "netflix-billing-secure.tk": "⚠️ This domain mimics Netflix to steal billing info.",
            "account-login-hdfcbank.xyz": "⚠️ This is a phishing domain spoofing HDFC Bank.",
            "amaz0n-verification.com": "⚠️ This domain tries to impersonate Amazon.",
            "ebay-security-check.store": "⚠️ This is a phishing domain pretending to be eBay.",
            "flipkart-update-center.co": "⚠️ Fake Flipkart update portal used for phishing.",
            "myntra-support-team.shop": "⚠️ Fake support domain pretending to be Myntra.",
            "olx-login-authentication.click": "⚠️ OLX login scam site used in phishing campaigns.",
            "googlerecovery-alerts.com": "⚠️ Fake Google recovery alert page.",
            "outlook-mail-security.net": "⚠️ Outlook phishing page aimed at email theft.",
            "icloud-login-confirm.gq": "⚠️ Phishing domain targeting iCloud users.",
            "yahoo-verification-service.ml": "⚠️ Yahoo login phishing site.",
            "zohoauth-reset-support.cf": "⚠️ Phishing site pretending to reset Zoho passwords.",
            "income-tax-india-verification.in": "⚠️ Fake Indian tax portal for identity theft.",
            "passport-update-portal.com": "⚠️ Fake passport update scam site.",
            "aadhaarcard-auth-gov.in.net": "⚠️ Fake Aadhaar authentication phishing domain.",
            "gov-panlink-update.click": "⚠️ PAN card phishing domain.",
            "epfo-securelogin-alerts.tk": "⚠️ Fake EPFO login phishing page.",
            "insta-login-helpdesk.cf": "⚠️ Instagram support phishing domain.",
            "facebook-recovery-security.gq": "⚠️ Fake Facebook recovery login page.",
            "whatsapp-backup-verification.ml": "⚠️ Phishing site targeting WhatsApp backups.",
            "snapchat-login-update.tk": "⚠️ Fake Snapchat login portal.",
            "twitter-security-check.gq": "⚠️ Twitter phishing site used in credential theft.",
            "microsoft-verify-account.cf": "⚠️ Microsoft credential phishing domain.",
            "gmail-alert-service.ml": "⚠️ Fake Gmail alert phishing link.",
            "linkedin-profile-auth.tk": "⚠️ Fake LinkedIn login phishing page.",
            "airtel-payment-warning.in": "⚠️ Airtel billing phishing site.",
            "jio-login-verify.xyz": "⚠️ Fake Jio user verification page.",
            "axis-secure-auth.com": "⚠️ Axis Bank phishing domain.",
            "icici-login-alert.net": "⚠️ Fake ICICI login warning site.",
            "sbi-kyc-update.co": "⚠️ KYC phishing targeting SBI users.",
            "yesbank-login-update.in": "⚠️ YES Bank spoof domain used in scams.",
            "pnb-security-verification.com": "⚠️ Fake Punjab National Bank login page.",
            "creditcard-service-check.cf": "⚠️ Fake credit card verification phishing domain.",
            "visa-auth-alert.tk": "⚠️ VISA phishing domain used for fraud.",
            "mastercard-security-check.ml": "⚠️ Fake Mastercard security update page.",
            "aadhaar-kyc-check.com": "⚠️ Aadhaar phishing domain used for ID theft.",
            "epf-verification-alerts.tk": "⚠️ Phishing site mimicking EPF portal.",
            "youtube1.com": "phishing url",
            
        }

        # Check for exact matches against known phishing domains
        domain = urlparse(url_input).hostname or url_input

        if domain in known_phishing_domains:
            return jsonify({
                'result': 'Phishing',
                'message': known_phishing_domains[domain]
            })

        # Proceed with normal model-based prediction
        features = extract_features_from_url(url_input)
        final_input = np.array([features])
        prediction = model.predict(final_input)[0]

        if prediction == 1:
            return jsonify({
                'result': 'Phishing',
                'message': f"⚠️ The domain '{url_input}' appears suspicious based on its structure."
            })
        else:
            return jsonify({
                'result': 'Legitimate',
                'message': f"✅ The domain '{url_input}' appears safe."
            })

    except Exception as e:
        return jsonify({'result': 'Error', 'message': f'Error: {str(e)}'})
