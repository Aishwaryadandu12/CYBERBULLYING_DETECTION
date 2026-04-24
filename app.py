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
    overflow: hidden;
}

/* Floating icons */
[data-testid="stAppViewContainer"]::after {
    content: "💬 📱 👍 ❤️ 🔁 📨 💻 🌐 📢 🧠";
    position: fixed;
    width: 200%;
    height: 200%;
    font-size: 40px;
    opacity: 0.08;
    animation: floatIcons 25s linear infinite;
}

/* Animation */
@keyframes floatIcons {
    0% { transform: translate(0,0); }
    50% { transform: translate(-200px,-200px); }
    100% { transform: translate(0,0); }
}

/* Overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.25);
}

/* White text */
* { color: white !important; }

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
    border-radius: 25px;
    padding: 10px 25px;
}
.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 20px #00c6ff;
}

/* Result box */
.result-box {
    margin-top: 20px;
    padding: 15px;
    border-radius: 12px;
}

/* Layout inside result */
.result-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}

/* Icon */
.result-icon {
    font-size: 32px;
    text-shadow: 0 0 10px white;
}

/* Text */
.result-text {
    font-size: 26px;
    font-weight: bold;
}

/* Safe */
.safe {
    background-color: rgba(0, 255, 150, 0.2) !important;
    border: 2px solid #00ffcc;
}

/* Cyberbullying */
.bad {
    background-color: rgba(255, 0, 0, 0.25) !important;
    border: 2px solid #ff4d4d;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚫 Cyberbullying Detection System")
st.write("AI-powered detection using Machine Learning (Random Forest + TF-IDF)")

# ---------------- INPUT ----------------
user_input = st.text_area("✍️ Enter Comment")

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a comment")
    else:
        label, score, cleaned = predict_comment(user_input)

        cyber_score = score * 100
        safe_score = (1 - score) * 100

        # RESULT DISPLAY WITH ICONS
        if "Cyberbullying" in label:
            st.markdown(f"""
            <div class="result-box bad">
                <div class="result-content">
                    <div class="result-icon">🚫</div>
                    <div class="result-text">
                        Cyberbullying ({cyber_score:.2f}%)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box safe">
                <div class="result-content">
                    <div class="result-icon">✅</div>
                    <div class="result-text">
                        Safe ({safe_score:.2f}%)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # METRICS
        col1, col2 = st.columns(2)
        col1.metric("🚫 Cyberbullying %", f"{cyber_score:.2f}%")
        col2.metric("✅ Safe %", f"{safe_score:.2f}%")

        # PROGRESS
        st.progress(int(cyber_score))

        # CLEANED TEXT
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
