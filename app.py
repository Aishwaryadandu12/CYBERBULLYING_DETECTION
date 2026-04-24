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

# ---------------- DEFAULT CSS ----------------
st.markdown("""
<style>

/* Default gradient before prediction */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e3c72, #2a5298, #6a11cb);
    transition: background 0.5s ease-in-out;
}

/* Gradient animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
            
/* 🔥 ICON BACKGROUND (force visible) */
.icon-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    font-size: 42px;
    opacity: 0.25;   /* HIGH visibility */
    z-index: 0;
    pointer-events: none;
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 25px;
    padding: 20px;
    animation: floatIcons 20s linear infinite;
}
/* Animation */
@keyframes floatIcons {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-40px); }
    100% { transform: translateY(0px); }
}
/* 🔥 FORCE APP ABOVE ICONS */
.block-container {
    position: relative;
    z-index: 2;
}
/* Glass UI */
.main {
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 18px;
}

/* Text color fix */
body, p, h1, h2, h3, h4, h5, h6, label, div, span {
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
    caret-color: #00f2fe !important;
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
/* 🔥 RANDOM FLOATING ICONS */
.icon-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
}

/* each icon random style */
.icon {
    position: absolute;
    font-size: 40px;
    opacity: 0.7;

    animation: floatRandom 6s ease-in-out infinite,
               glowPulse 2s ease-in-out infinite alternate;

    text-shadow:
        0 0 5px #00f2fe,
        0 0 10px #00f2fe,
        0 0 20px #00f2fe,
        0 0 40px #0072ff;
}

/* floating animation */
@keyframes floatRandom {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
    100% { transform: translateY(0px); }
}

/* glow */
@keyframes glowPulse {
    from { opacity: 0.4; }
    to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# 🔥 ICON BACKGROUND (FULL COVERAGE)
st.markdown("""
<div class="icon-bg">

<span class="icon" style="top:5%; left:10%;">📱</span>
<span class="icon" style="top:20%; left:80%;">💬</span>
<span class="icon" style="top:40%; left:30%;">❤️</span>
<span class="icon" style="top:70%; left:60%;">👍</span>
<span class="icon" style="top:85%; left:15%;">🔁</span>
<span class="icon" style="top:60%; left:85%;">📨</span>
<span class="icon" style="top:15%; left:50%;">🌐</span>
<span class="icon" style="top:35%; left:70%;">📢</span>
<span class="icon" style="top:55%; left:25%;">💻</span>
<span class="icon" style="top:75%; left:45%;">📸</span>
<span class="icon" style="top:10%; left:75%;">📲</span>
<span class="icon" style="top:90%; left:35%;">🔔</span>

<span class="icon" style="top:25%; left:5%;">💬</span>
<span class="icon" style="top:50%; left:50%;">❤️</span>
<span class="icon" style="top:65%; left:10%;">👍</span>
<span class="icon" style="top:80%; left:80%;">📱</span>
<span class="icon" style="top:30%; left:60%;">📢</span>
<span class="icon" style="top:45%; left:15%;">💻</span>

</div>
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

        # 🔥 FULL PAGE BACKGROUND CHANGE
        if "Cyberbullying" in label:
            st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #3b0000, #8b0000, #ff0000) !important;
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #003b1f, #007f3f, #00ff88) !important;
            }
            </style>
            """, unsafe_allow_html=True)

        # RESULT TEXT
        if "Cyberbullying" in label:
            st.error(f"🚫 Cyberbullying ({cyber_score:.2f}%)")
        else:
            st.success(f"✅ Safe ({safe_score:.2f}%)")

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
