import streamlit as st
import joblib

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Performance Predictor",
    page_icon="🏏",
    layout="centered"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.stApp {
    background: #030d07;
}

/* Animation fix */
.prob-fill {
    height: 100%;
    border-radius: 100px;
    animation: growBar 1s ease forwards;
}
@keyframes growBar {
    from { width: 0; }
    to { width: 100%; }
}

/* (rest of your CSS unchanged) */
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown("""
<div class="pitch-hero">
    <h1 class="hero-title">Cricket <em>Performance</em> Predictor</h1>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load("cricket_model.pkl")
    scaler = joblib.load("cricket_scaler.pkl")
    return model, scaler

try:
    model_ml, scaler = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found: {e}")

# ─────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    Ave = st.number_input("Average", 0.0, value=30.0)
    century_rate = st.number_input("Century Rate", 0.0, value=0.0)
    duck_rate = st.number_input("Duck Rate", 0.0, value=3.0)
    Mat = st.number_input("Matches", 1, value=50)

with col2:
    HS = st.number_input("Highest Score", 0, value=90)
    fifty_rate = st.number_input("Fifty Rate", 0.0, value=5.0)
    no_rate = st.number_input("Not-Out %", 0.0, 100.0, value=10.0)

# ─────────────────────────────────────────
# Predict
# ─────────────────────────────────────────
TIER_CONFIG = {
    "Poor":    {"color": "#f87171", "bar": "linear-gradient(90deg,#f87171,#ef4444)"},
    "Average": {"color": "#60a5fa", "bar": "linear-gradient(90deg,#60a5fa,#3b82f6)"},
    "Elite":   {"color": "#4ade80", "bar": "linear-gradient(90deg,#4ade80,#22c55e)"},
}

if st.button("Analyse Player"):
    if not model_loaded:
        st.stop()

    sample = [[Ave, HS, century_rate, fifty_rate, duck_rate, no_rate, Mat]]
    sample_scaled = scaler.transform(sample)

    pred_class = model_ml.predict(sample_scaled)[0]
    pred_proba = model_ml.predict_proba(sample_scaled)[0]

    confidence = float(max(pred_proba)) * 100

    st.markdown(f"### Prediction: **{pred_class}** ({confidence:.1f}%)")

    # ── FIXED probability mapping ──
    class_probs = dict(zip(model_ml.classes_, pred_proba))

    prob_rows = ""
    for cls in ["Poor", "Average", "Elite"]:
        prob = class_probs.get(cls, 0)
        c = TIER_CONFIG[cls]
        pct = prob * 100

        prob_rows += f"""
        <div style="display:flex;align-items:center;margin-bottom:10px;">
            <div style="width:80px;">{cls}</div>
            <div style="flex:1;background:#111;height:8px;border-radius:10px;">
                <div class="prob-fill" style="width:{pct:.1f}%;background:{c['bar']};height:100%;border-radius:10px;"></div>
            </div>
            <div style="width:50px;text-align:right;">{pct:.1f}%</div>
        </div>
        """

    st.markdown(prob_rows, unsafe_allow_html=True)
