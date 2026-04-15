import pickle
import re
from pathlib import Path

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
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------------- PREDICTION ----------------
def predict_comment(text, threshold=0.35):
    cleaned = clean_text(text)

    # 🔥 Rule-based boost (important for accuracy)
    bad_words = ["stupid", "idiot", "useless", "hate", "fool", "dumb"]
    if any(word in cleaned for word in bad_words):
        return "Cyberbullying 🚫", 0.95, cleaned

    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0][1]

    label = "Cyberbullying 🚫" if prob >= threshold else "Safe ✅"
    return label, prob, cleaned

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🚫",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* 🌈 Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e3c72, #2a5298, #6a11cb);
}

/* Overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.25);
    z-index: 0;
}

/* Force white text */
* {
    color: white !important;
}

/* Title */
h1 {
    text-align: center;
    font-size: 48px;
    text-shadow: 0 0 18px #00f2fe;
}

/* Input */
textarea {
    background: rgba(0,0,0,0.6) !important;
    color: #00f2fe !important;
    border-radius: 12px !important;
    border: 1px solid #00c6ff !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border-radius: 25px;
    padding: 10px 25px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 20px #00c6ff;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 30px !important;
    font-weight: bold !important;
    color: #00f2fe !important;
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
}

/* Result styles */
.result-box {
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    margin-top: 15px;
    padding: 12px;
}

.safe {
    color: #00ffcc !important;
}

.bad {
    color: #ff4d4d !important;
}

/* Code box */
.stCodeBlock {
    background: rgba(0,0,0,0.6) !important;
    color: #00f2fe !important;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚫 Cyberbullying Detection System")
st.write("AI-powered detection using Machine Learning ")

# ---------------- INPUT ----------------
user_input = st.text_area(
    "✍️ Enter Comment",
    placeholder="Type a message to analyze..."
)

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a comment")
    else:
        label, score, cleaned = predict_comment(user_input)

        cyber_score = score * 100
        safe_score = (1 - score) * 100

        # RESULT DISPLAY
        if "Cyberbullying" in label:
            st.markdown(f"""
            <div class="result-box bad">
                🚫 {label} ({cyber_score:.2f}%)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box safe">
                ✅ {label} ({safe_score:.2f}%)
            </div>
            """, unsafe_allow_html=True)

        # METRICS
        col1, col2 = st.columns(2)
        col1.metric("🚫 Cyberbullying %", f"{cyber_score:.2f}%")
        col2.metric("✅ Safe %", f"{safe_score:.2f}%")

        # PROGRESS BAR
        st.progress(int(cyber_score))

        # CLEANED TEXT (for demo clarity)
        with st.expander("🔍 See Processed Text"):
            st.write(cleaned)

# ---------------- EXAMPLES ----------------
st.markdown("""
### 💡 Try Examples:
- You are amazing and kind
- You are stupid and useless
- I hate you
- Great work, keep going!
""")
