# phishing-url-and-email-detection-using-machine-learning
ClickShield is a real-time phishing detection system that combines machine learning and NLP to identify malicious URLs and emails with high accuracy. Designed with a fully interactive and modern web interface, it helps users stay protected from cyber threats using advanced AI techniques.

# 🛡️ ClickShield: AI-Based Phishing URL & Email Detection Web App

ClickShield is a real-time phishing detection system that analyzes both **URLs** and **emails** using machine learning and natural language processing. It features a fully interactive **Flask web interface** with modern design elements, including video headers and animated phishing awareness tips.

---

## 🚀 Features

### 🔗 Phishing URL Detection
- Real-time prediction of phishing websites
- Feature engineering includes:
  - URL entropy
  - Suspicious top-level domains (TLDs)
  - Use of IPs instead of domains
  - Redirection symbols (`//`, `@`, etc.)
  - Suspicious keywords (e.g., "login", "verify", "secure")
  - Domain repetition & hyphenation
  - Length and character frequency analysis
- Numeric features are standardized using `StandardScaler`
- PCA is applied to reduce `Entropy` and `Length` to a single component
- Trained using a **tuned XGBoost model**
- Includes a hardcoded list of verified phishing domains

### 📧 Phishing Email Detection
- Users can upload `.txt` files or paste content
- Uses NLP techniques to analyze emails:
  - `TfidfVectorizer` on email body
  - Trained on labeled datasets
  - Works with both **Keras (LSTM)** or **Scikit-learn (Logistic Regression/SVM)** models
- Detects phishing phrases, tone, and style

---

## 🧠 ML Pipeline

### URL Detection Pipeline:
1. Extract custom features via `feature.py`
2. Apply `StandardScaler` to numeric fields
3. Reduce entropy/length features using `PCA`
4. Predict using trained **XGBoost** classifier
5. Check against a **hardcoded blacklist** for known phishing domains

### Email Detection Pipeline:
1. Preprocess and clean text
2. Vectorize using `TfidfVectorizer`
3. Predict via pre-trained ML model (LSTM or traditional)
4. Display probability of phishing with friendly UI feedback

---

## 🖼️ UI Preview

| Home Page | URL Detection | Email Detection |
|-----------|---------------|-----------------|
| ![home](screenshots/home.png) | ![url](screenshots/url.png) | ![email](screenshots/email.png) |

---

## 📁 Project Structure

phishing-detector/
│
├── run.py # Main entry point for the web app
├── url_folder/
│ ├── app.py # URL detection backend
│ ├── feature.py # Custom feature extractor for URLs
│ ├── phishing_urls.txt # Hardcoded phishing domains
│
├── email_folder/
│ ├── email.py # Email detection backend
│ ├── templates/email.html
│
├── models/
│ ├── model.pkl # XGBoost model (URLs)
│ ├── scaler.pkl # StandardScaler
│ ├── pca.pkl # PCA transformer
│ ├── email_model.pkl # ML/NLP model for emails
│ ├── vectorizer.pkl # TF-IDF vectorizer for email detection
│
├── templates/
│ ├── index.html # Homepage with animated video UI
│
├── static/ # CSS, JS, videos, animations
├── requirements.txt # Python dependencies
└── README.md
