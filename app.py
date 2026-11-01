import re
import ipaddress
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from database import create_table, save_detection, get_history, delete_history
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Suppress warnings from scikit-learn
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version mismatch warnings from scikit-learn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

create_table()

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ScamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #080d18;
    --panel: #101a2b;
    --panel-soft: #142238;
    --border: rgba(148, 163, 184, .16);
    --cyan: #35d7e8;
    --text: #edf5ff;
    --muted: #91a4bb;
}
.stApp {
    background:
      radial-gradient(ellipse at 82% 0%, rgba(21, 113, 145, .18), transparent 34%),
      radial-gradient(ellipse at 0% 35%, rgba(49, 76, 150, .13), transparent 32%),
      var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: rgba(8,13,24,.72); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e192a 0%, #0a1220 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div { padding-top: 1.6rem; }
h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -.035em;
}
h1 { font-size: 2.45rem !important; font-weight: 700 !important; }
h2 { font-size: 1.65rem !important; }
p, label, [data-testid="stMarkdownContainer"] { color: #c1cede; }
small, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(20,34,56,.96), rgba(13,24,41,.96));
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 20px 22px;
    box-shadow: 0 12px 35px rgba(0,0,0,.14);
}
[data-testid="stMetricLabel"] { color: #9fb2c9 !important; }
[data-testid="stMetricValue"] { color: #f3f8ff !important; font-family: 'Space Grotesk', sans-serif; }

.stButton > button {
    box-sizing: border-box;
    width: 100%;
    min-height: 2.8rem;
    padding: .65rem 1rem;
    border: 1px solid transparent !important;
    border-radius: 11px;
    background: linear-gradient(105deg, #23bfd4, #48e0d0) !important;
    color: #06121e !important;
    -webkit-text-fill-color: #06121e;
    font-weight: 700;
    line-height: 1.2;
    white-space: normal;
    overflow-wrap: anywhere;
    transition: box-shadow .2s ease, filter .2s ease;
    box-shadow: 0 7px 22px rgba(35,191,212,.16);
}
.stButton > button:hover:not(:disabled) {
    filter: brightness(1.06);
    box-shadow: 0 10px 28px rgba(35,191,212,.28);
}
.stButton > button:focus-visible {
    outline: 3px solid rgba(53,215,232,.65);
    outline-offset: 3px;
}
.stButton > button:active:not(:disabled) { filter: brightness(.94); }
.stButton > button:disabled {
    cursor: not-allowed;
    opacity: .58;
    box-shadow: none;
}
.stTextInput input, .stTextArea textarea {
    background: #0d192a !important;
    color: #f2f7ff !important;
    border: 1px solid #2a3b53 !important;
    border-radius: 11px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 1px var(--cyan) !important;
}
[data-testid="stRadio"] label { color: #d7e3f1 !important; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #24bfd4, #61e4c5);
}
[data-testid="stAlert"] { border-radius: 13px; }
hr { border-color: var(--border) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 13px;
    overflow: hidden;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(16,26,43,.58);
    border: 1px solid var(--border);
    border-radius: 17px;
}
div[data-testid="stSidebar"] .stRadio > label {
    color: #91a4bb !important;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .12em;
}
div[data-testid="stSidebar"] [role="radiogroup"] {
    gap: .35rem;
}
div[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: .65rem .8rem;
    border-radius: 10px;
}
div[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(53,215,232,.09);
}
div[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(53,215,232,.13);
    color: #7ce8ef !important;
}
div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #9db0c7;
}
[data-testid="stHorizontalBlock"] { gap: 1rem; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1440px; }
@media (max-width: 640px) {
    .block-container { padding: 1rem 1rem 1.5rem; }
    h1 { font-size: 1.9rem !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:12px;padding:13px 18px;margin:0 0 22px 0;border:1px solid rgba(148,163,184,.16);
            border-radius:15px;background:rgba(16,26,43,.72);">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:39px;height:39px;border-radius:12px;display:grid;place-items:center;
                background:linear-gradient(135deg,#20bfd5,#54e2c8);font-size:21px;">🛡️</div>
    <div>
      <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#f3f8ff;
                  font-size:17px;letter-spacing:-.4px;">SCAMSHIELD <span style="color:#35d7e8;">AI</span></div>
      <div style="font-size:10px;letter-spacing:2px;color:#91a4bb;">INTELLIGENT THREAT DETECTION</div>
    </div>
  </div>
  <div style="font-size:11px;color:#8da4bb;border:1px solid rgba(53,215,232,.25);
              padding:7px 11px;border-radius:30px;">● SYSTEM READY</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():
    model_paths = {
        "sms": MODEL_DIR / "scamshield_model.pkl",
        "job": MODEL_DIR / "fake_job_model.pkl",
        "url": MODEL_DIR / "url_model.pkl",
    }

    missing = [path.name for path in model_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model file not found: " + ", ".join(missing)
        )

    sms = joblib.load(model_paths["sms"])
    job = joblib.load(model_paths["job"])
    url = joblib.load(model_paths["url"])

    return sms, job, url


try:
    sms_model, job_model, url_model = load_models()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


# ==========================================
# DATABASE HELPERS
# ==========================================

def save_history(module, result, confidence):
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    rounded_conf = round(confidence, 2)
    save_detection(module, result, rounded_conf, current_time)

def get_normalized_history():
    raw_records = get_history()
    normalized = []
    for item in raw_records:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, (tuple, list)):
            if len(item) >= 5:
                normalized.append({
                    "id": item[0],
                    "Module": item[1],
                    "Result": item[2],
                    "Confidence": item[3],
                    "Time": item[4]
                })
            elif len(item) == 4:
                normalized.append({
                    "Module": item[0],
                    "Result": item[1],
                    "Confidence": item[2],
                    "Time": item[3]
                })
    return normalized


# ==========================================
# URL FEATURE EXTRACTION
# ==========================================

def extract_url_features(url):
    url = url.strip()

    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    domain = parsed.hostname or ""

    try:
        ipaddress.ip_address(domain)
        is_ip = 1
    except ValueError:
        is_ip = 0

    digit_count = sum(c.isdigit() for c in url)
    letter_count = sum(c.isalpha() for c in url)

    special_count = len(
        re.findall(r"[^A-Za-z0-9]", url)
    )

    domain_parts = domain.split(".") if domain else []
    tld = domain_parts[-1] if domain_parts else ""

    subdomain_count = max(len(domain_parts) - 2, 0)

    suspicious_words = [
        "login", "verify", "update", "secure",
        "account", "banking", "confirm",
        "password", "signin", "paypal"
    ]

    has_suspicious_word = int(
        any(word in url.lower() for word in suspicious_words)
    )

    return {
        "url_length": len(url),
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


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:
    st.markdown('<h2 style="margin-bottom:0;color:#f3f8ff">🛡️ ScamShield AI</h2>', unsafe_allow_html=True)
    st.caption("YOUR AI-POWERED SECURITY CENTER")
    st.divider()

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    page_options = [
        "Dashboard",
        "SMS Detection",
        "Fake Job Detection",
        "URL Detection",
        "Detection History"
    ]
    
    current_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0

    page = st.radio(
        "NAVIGATION",
        page_options,
        index=current_index,
        key="page"
    )

    st.divider()
    st.markdown('<div style="padding:10px 12px;border:1px solid rgba(53,215,232,.22);border-radius:11px;background:rgba(53,215,232,.07);color:#74e5d7;font-size:12px;font-weight:600;">●  3 AI MODELS CONNECTED</div>', unsafe_allow_html=True)
    st.caption("Python • Scikit-learn • Streamlit • SQLite")


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":
    st.markdown("<div style='color:#35d7e8;font-size:11px;font-weight:700;letter-spacing:2px;margin-bottom:7px;'>OVERVIEW / CONTROL CENTER</div>", unsafe_allow_html=True)
    st.title("Security Dashboard")

    st.write(
        "Welcome to ScamShield AI. Monitor and analyse "
        "potential online threats."
    )

    st.divider()

    history_records = get_normalized_history()
    total = len(history_records)

    suspicious = sum(
        1 for item in history_records
        if item["Result"] in [
            "Spam",
            "Fraudulent Job",
            "Phishing URL"
        ]
    )

    safe = total - suspicious

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Analyses", total)
    col2.metric("Suspicious Results", suspicious)
    col3.metric("Normal Results", safe)

    st.subheader("Detection Modules")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📩 **SMS Detection**")
        st.write("Identify suspicious text messages.")
        st.caption("Machine Learning: TF-IDF + Logistic Regression")

    with col2:
        st.info("💼 **Fake Job Detection**")
        st.write("Analyse job descriptions for possible fraud.")
        st.caption("Machine Learning: TF-IDF + Logistic Regression")

    with col3:
        st.info("🔗 **URL Detection**")
        st.write("Identify potentially malicious website links.")
        st.caption("Machine Learning: Logistic Regression")

    st.divider()

    st.subheader("Recent Activity")

    if history_records:
        recent = pd.DataFrame(history_records[-5:][::-1])
        st.dataframe(
            recent,
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No detection activity yet. Try one of the detection modules.")


# ==========================================
# SMS DETECTION
# ==========================================

elif page == "SMS Detection":
    st.markdown("<div style='color:#35d7e8;font-size:11px;font-weight:700;letter-spacing:2px;margin-bottom:7px;'>THREAT ANALYSIS / MESSAGES</div>", unsafe_allow_html=True)
    st.title("📩 SMS Scam Detection")

    st.write("Analyse a message to identify possible spam.")

    message = st.text_area(
        "Enter your message",
        placeholder="Congratulations! You won a free prize...",
        height=160
    )

    if st.button("Analyse Message", width="stretch"):
        if message.strip():
            prediction = sms_model.predict([message])[0]
            probabilities = sms_model.predict_proba([message])[0]

            if prediction == 1:
                result = "Spam"
                confidence = probabilities[1] * 100
                st.error("⚠️ Suspicious message detected!")
            else:
                result = "Normal"
                confidence = probabilities[0] * 100
                st.success("✅ Normal-like message detected.")

            st.metric("Model Confidence", f"{confidence:.2f}%")
            st.progress(float(confidence / 100))

            save_history("SMS", result, confidence)
            st.caption("AI predictions are not a guarantee of message safety.")
        else:
            st.warning("Please enter a message.")


# ==========================================
# FAKE JOB DETECTION
# ==========================================

elif page == "Fake Job Detection":
    st.markdown("<div style='color:#35d7e8;font-size:11px;font-weight:700;letter-spacing:2px;margin-bottom:7px;'>THREAT ANALYSIS / EMPLOYMENT</div>", unsafe_allow_html=True)
    st.title("💼 Fake Job Detection")

    st.write("Analyse job details to identify potential fraud.")

    title = st.text_input("Job Title")
    company = st.text_area("Company Profile")
    description = st.text_area("Job Description")
    requirements = st.text_area("Requirements")
    benefits = st.text_area("Benefits")

    if st.button("Analyse Job Posting", width="stretch"):
        job_text = " ".join([
            title, company, description,
            requirements, benefits
        ])

        if job_text.strip():
            prediction = job_model.predict([job_text])[0]
            probabilities = job_model.predict_proba([job_text])[0]

            if prediction == 1:
                result = "Fraudulent Job"
                confidence = probabilities[1] * 100
                st.error("⚠️ Potentially fraudulent job posting!")
            else:
                result = "Legitimate Job"
                confidence = probabilities[0] * 100
                st.success("✅ Legitimate-like job posting.")

            st.metric("Model Confidence", f"{confidence:.2f}%")
            st.progress(float(confidence / 100))

            save_history("Fake Job", result, confidence)
            st.caption("Always independently verify the employer and job offer.")
        else:
            st.warning("Please enter job posting details.")


# ==========================================
# URL DETECTION
# ==========================================

elif page == "URL Detection":
    st.markdown("<div style='color:#35d7e8;font-size:11px;font-weight:700;letter-spacing:2px;margin-bottom:7px;'>THREAT ANALYSIS / WEB LINKS</div>", unsafe_allow_html=True)
    st.title("🔗 Phishing URL Detection")

    st.write("Analyse URL characteristics for possible phishing.")

    url = st.text_input(
        "Enter website URL",
        placeholder="https://example.com"
    )

    if st.button("Analyse URL", width="stretch"):
        if url.strip():
            normalized_url = url.strip()

            if "://" not in normalized_url:
                normalized_url = "https://" + normalized_url

            try:
                parsed = urlparse(normalized_url)
                domain = parsed.hostname or ""
                port = parsed.port
                valid_url = (
                    parsed.scheme.lower() in ["http", "https"]
                    and bool(domain)
                    and "." in domain
                    and not any(
                        character.isspace()
                        for character in normalized_url
                    )
                    and (port is None or 1 <= port <= 65535)
                )
            except ValueError:
                valid_url = False

            if not valid_url:
                st.error("Please enter a valid website URL.")
            else:
                features = extract_url_features(normalized_url)
                input_data = pd.DataFrame([features])

                prediction = url_model.predict(input_data)[0]
                probabilities = url_model.predict_proba(input_data)[0]

                if prediction == 0:
                    result = "Phishing URL"
                    confidence = probabilities[0] * 100
                    st.error("⚠️ Potential phishing URL detected!")
                else:
                    result = "Legitimate URL"
                    confidence = probabilities[1] * 100
                    st.success("✅ This resembles a legitimate URL.")

                st.metric("Model Confidence", f"{confidence:.2f}%")
                st.progress(float(confidence / 100))

                save_history("URL", result, confidence)

                st.warning(
                    "This is an AI prediction, not a safety guarantee. "
                    "Do not enter sensitive information on unverified websites."
                )
        else:
            st.warning("Please enter a URL.")


# ==========================================
# DETECTION HISTORY
# ==========================================

elif page == "Detection History":
    st.markdown("<div style='color:#35d7e8;font-size:11px;font-weight:700;letter-spacing:2px;margin-bottom:7px;'>ACTIVITY / AUDIT LOG</div>", unsafe_allow_html=True)
    st.title("📊 Detection History")

    st.write("Review your previous detection results.")

    history_records = get_normalized_history()

    if history_records:
        history_df = pd.DataFrame(history_records[::-1])

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True
        )

        st.subheader("Detection Summary")

        chart_data = (
            pd.DataFrame(history_records)
            ["Module"]
            .value_counts()
        )

        st.bar_chart(chart_data)

        if st.button("Clear Detection History", width="stretch"):
            delete_history()
            st.rerun()
    else:
        st.info("No detection history available yet.")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "🛡️ ScamShield AI | AI-powered scam detection platform | "
    "Educational project. Predictions are not security guarantees."
)