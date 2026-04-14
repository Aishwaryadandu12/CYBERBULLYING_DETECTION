import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cyberbullying_rf_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

# ---------------- TEXT CLEANING ----------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
        return model, vectorizer

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ---------------- PREDICTION ----------------
def predict_comment(comment: str, threshold: float = 0.5):
    model, vectorizer = load_artifacts()

    if model is None or vectorizer is None:
        return "Error", 0.0, ""

    cleaned = clean_text(comment)

    vec = vectorizer.transform([cleaned])

    try:
        score = float(model.predict_proba(vec)[0][1])
    except:
        score = 0.0

    label = "Cyberbullying 🚫" if score >= threshold else "Safe ✅"

    return label, score, cleaned

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cyberbullying Detection Dashboard",
    page_icon="🚫",
    layout="wide"
)

# ---------------- YOUR EXISTING CSS (UNCHANGED) ----------------
st.markdown("""
<style>

/* 🌈 BRIGHT BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e3c72, #2a5298, #6a11cb);
    overflow: hidden;
}

/* 🔥 Floating Icons */
.icon {
    position: fixed;
    width: 70px;
    opacity: 0.25;
    animation: floatUp 18s linear infinite;
    z-index: 0;
}

.icon:nth-child(1) { left: 10%; animation-delay: 0s; }
.icon:nth-child(2) { left: 30%; animation-delay: 5s; }
.icon:nth-child(3) { left: 60%; animation-delay: 10s; }
.icon:nth-child(4) { left: 80%; animation-delay: 15s; }

@keyframes floatUp {
    0% { bottom: -100px; transform: translateX(0) rotate(0deg); }
    50% { transform: translateX(40px) rotate(180deg); }
    100% { bottom: 110%; transform: translateX(-40px) rotate(360deg); }
}

/* LIGHT OVERLAY */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    z-index: 0;
}

[data-testid="stAppViewContainer"] > * {
    position: relative;
    z-index: 1;
}

/* TITLE */
h1 {
    text-align: center;
    font-size: 50px;
    color: #ffffff;
    text-shadow:
        0 0 10px #00f2fe,
        0 0 20px #4facfe,
        0 0 30px #00c6ff;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border-radius: 25px;
    padding: 10px 25px;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.1);
    box-shadow: 0 0 20px #00c6ff;
}

</style>

<!-- FLOATING ICONS -->
<img class="icon" src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png">
<img class="icon" src="https://cdn-icons-png.flaticon.com/512/733/733547.png">
<img class="icon" src="https://cdn-icons-png.flaticon.com/512/733/733579.png">
<img class="icon" src="https://cdn-icons-png.flaticon.com/512/733/733558.png">
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.title("🚫 Cyberbullying Detection System")
st.write("Analyze comments using Machine Learning (Random Forest)")

# ---------------- FILE CHECK ----------------
if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.error("❌ Missing model files!")
    st.code("""
cyberbullying_rf_model.pkl
tfidf_vectorizer.pkl
    """)
    st.stop()

# ---------------- INPUT ----------------
user_input = st.text_area(
    "",
    placeholder="Type your comment here..."
)

# ---------------- ANALYZE ----------------
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Enter a comment")
    else:
        label, score, cleaned = predict_comment(user_input)

        cyber_score = score * 100
        safe_score = (1 - score) * 100

        # RESULT
        if "Cyberbullying" in label:
            st.error(f"{label} ({cyber_score:.2f}%)")
        else:
            st.success(f"{label} ({safe_score:.2f}%)")

        # METRICS
        m1, m2 = st.columns(2)
        m1.metric("🚫 Cyberbullying %", f"{cyber_score:.2f}%")
        m2.metric("✅ Safe %", f"{safe_score:.2f}%")

        # PROGRESS
        st.progress(int(cyber_score))

        # CLEANED TEXT
        st.subheader("🔍 Processed Text")
        st.code(cleaned)

# ---------------- EXAMPLES ----------------
st.markdown("""
💡 Try these examples:

✔ You did a great job!  
⚠ You are dumb  
⚠ You are stupid  
""")