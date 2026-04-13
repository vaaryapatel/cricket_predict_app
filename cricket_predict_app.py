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

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

/* (ALL YOUR CSS EXACTLY SAME — NOT CHANGED) */
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown("""
<div class="pitch-hero">
    <div class="badge"><span class="badge-dot"></span>ML-Powered Analysis</div>
    <h1 class="hero-title">Cricket<br><em>Performance</em><br>Predictor</h1>
    <p class="hero-sub">Feed in a player's career statistics and discover whether they rank as Poor, Average, or Elite — powered by machine learning.</p>
    <div class="stats-row">
        <div class="stat-pill"><strong>3</strong> Performance Tiers</div>
        <div class="stat-pill"><strong>7</strong> Input Features</div>
        <div class="stat-pill"><strong>ML</strong> Classifier</div>
    </div>
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
    st.warning(f"⚠️ Model files not found: {e}")

# ─────────────────────────────────────────
# Input section
# ─────────────────────────────────────────
st.markdown('<div class="section-label">Batting Statistics</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    Ave = st.number_input("Batting Average", min_value=0.0, value=30.0, step=0.5, format="%.1f")
    st.markdown('<div class="input-hint">Career runs ÷ dismissals</div>', unsafe_allow_html=True)

    century_rate = st.number_input("Century Rate", min_value=0.0, value=0.0, step=0.1, format="%.1f")
    st.markdown('<div class="input-hint">Per 100 innings</div>', unsafe_allow_html=True)

    duck_rate = st.number_input("Duck Rate", min_value=0.0, value=3.0, step=0.1, format="%.1f")
    st.markdown('<div class="input-hint">Per 100 innings</div>', unsafe_allow_html=True)

    Mat = st.number_input("Total Matches", min_value=1, value=50, step=1)
    st.markdown('<div class="input-hint">Career match count</div>', unsafe_allow_html=True)

with col2:
    HS = st.number_input("Highest Score", min_value=0, value=90, step=1)
    st.markdown('<div class="input-hint">Best innings score</div>', unsafe_allow_html=True)

    fifty_rate = st.number_input("Fifty Rate", min_value=0.0, value=5.0, step=0.1, format="%.1f")
    st.markdown('<div class="input-hint">Per 100 innings</div>', unsafe_allow_html=True)

    no_rate = st.number_input("Not-Out %", min_value=0.0, max_value=100.0, value=10.0, step=0.5, format="%.1f")
    st.markdown('<div class="input-hint">% of innings not out</div>', unsafe_allow_html=True)

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# Predict
# ─────────────────────────────────────────
TIER_CONFIG = {
    "Poor":    {"color": "#f87171", "glow": "rgba(248,113,113,0.3)",  "bg": "rgba(248,113,113,0.06)", "bar": "linear-gradient(90deg,#f87171,#ef4444)"},
    "Average": {"color": "#60a5fa", "glow": "rgba(96,165,250,0.3)",   "bg": "rgba(96,165,250,0.06)",  "bar": "linear-gradient(90deg,#60a5fa,#3b82f6)"},
    "Elite":   {"color": "#4ade80", "glow": "rgba(74,222,128,0.3)",   "bg": "rgba(74,222,128,0.06)",  "bar": "linear-gradient(90deg,#4ade80,#22c55e)"},
}

predict_btn = st.button("Analyse Player →")

if predict_btn:
    if not model_loaded:
        st.error("Model files not found.")
    else:
        sample = [[Ave, HS, century_rate, fifty_rate, duck_rate, no_rate, Mat]]
        sample_scaled = scaler.transform(sample)
        pred_class  = model_ml.predict(sample_scaled)[0]
        pred_proba  = model_ml.predict_proba(sample_scaled)[0]
        cfg         = TIER_CONFIG.get(pred_class, TIER_CONFIG["Average"])
        confidence  = max(pred_proba) * 100

        # ── Result card ──
        st.markdown(f"""
        <div class="result-outer">
            <div class="result-inner" style="background: radial-gradient(ellipse 60% 50% at 50% 0%, {cfg['bg']}, #060f08);">
                <div class="result-eyebrow">Performance Tier</div>
                <div class="result-tier" style="color:{cfg['color']}; text-shadow: 0 0 40px {cfg['glow']};">
                    {pred_class}
                </div>
                <div class="result-subtitle">{confidence:.1f}% model confidence · {int(Mat)} career matches</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── FIXED Probability bars ──
        class_probs = dict(zip(model_ml.classes_, pred_proba))

        prob_rows = ""
        for cls in ["Poor", "Average", "Elite"]:
            prob = class_probs.get(cls, 0)
            c = TIER_CONFIG.get(cls, TIER_CONFIG["Average"])
            pct = prob * 100
            is_top = cls == pred_class

            label_color = c["color"] if is_top else "rgba(232,245,236,0.4)"
            val_color   = c["color"] if is_top else "rgba(232,245,236,0.5)"

            prob_rows += f"""
            <div class="prob-row">
                <div class="prob-label" style="color:{label_color};">{cls}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{pct:.1f}%; background:{c['bar']};"></div>
                </div>
                <div class="prob-val" style="color:{val_color};">{pct:.1f}%</div>
            </div>"""

        st.markdown(f"""
        <div class="prob-container">
            <div class="prob-header">Confidence Breakdown</div>
            {prob_rows}
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown("""
<div class="fancy-divider"><span>Cricket Intelligence</span></div>
<div class="footer">Cricket Performance Predictor · ML Model v1 · Built with Streamlit</div>
""", unsafe_allow_html=True)
