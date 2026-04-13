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

/* ── Animated gradient background ── */
.stApp {
    background: #030d07;
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(16,110,48,0.45) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 85% 80%, rgba(5,60,25,0.3) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 10% 90%, rgba(8,80,35,0.25) 0%, transparent 60%);
    min-height: 100vh;
}

/* ── Container ── */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 680px !important;
}

/* ── Cricket pitch SVG hero ── */
.pitch-hero {
    position: relative;
    padding: 4rem 1.5rem 2.5rem;
    text-align: center;
    overflow: hidden;
}
.pitch-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='600' height='200' viewBox='0 0 600 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cellipse cx='300' cy='200' rx='240' ry='80' fill='none' stroke='rgba(255,255,255,0.03)' stroke-width='1'/%3E%3Cellipse cx='300' cy='200' rx='180' ry='60' fill='none' stroke='rgba(255,255,255,0.04)' stroke-width='1'/%3E%3Cellipse cx='300' cy='200' rx='120' ry='40' fill='none' stroke='rgba(255,255,255,0.05)' stroke-width='1'/%3E%3Crect x='275' y='120' width='50' height='80' fill='none' stroke='rgba(255,255,255,0.07)' stroke-width='1'/%3E%3C/svg%3E") center bottom no-repeat;
    pointer-events: none;
    opacity: 0.7;
}

/* ── Badge ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52,199,89,0.12);
    border: 1px solid rgba(52,199,89,0.25);
    color: #34c759;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    padding: 5px 16px 5px 12px;
    border-radius: 100px;
    margin-bottom: 1.4rem;
}
.badge-dot {
    width: 6px; height: 6px;
    background: #34c759;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}

/* ── Hero title ── */
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 700;
    color: #e8f5ec;
    line-height: 1.05;
    margin: 0 0 1rem;
    letter-spacing: -0.01em;
}
.hero-title em {
    font-style: italic;
    color: #4ade80;
}
.hero-sub {
    color: rgba(232,245,236,0.45);
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 440px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
}

/* ── Stats pills row ── */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 2.5rem;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.72rem;
    color: rgba(232,245,236,0.45);
    letter-spacing: 0.06em;
}
.stat-pill strong {
    color: #4ade80;
    font-weight: 600;
}

/* ── Glass card wrapper ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 1.2rem;
}

/* ── Section label ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(74,222,128,0.2), transparent);
}

/* ── Streamlit number input overrides ── */
.stNumberInput > label {
    color: rgba(232,245,236,0.5) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.stNumberInput input {
    background: #0d2016 !important;
    border: 1px solid rgba(74,222,128,0.22) !important;
    border-radius: 12px !important;
    color: #e8f5ec !important;
    -webkit-text-fill-color: #e8f5ec !important;
    caret-color: #4ade80 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
}
.stNumberInput input:focus {
    background: #112a1a !important;
    border-color: rgba(74,222,128,0.55) !important;
    box-shadow: 0 0 0 3px rgba(74,222,128,0.1) !important;
    -webkit-text-fill-color: #e8f5ec !important;
}
.stNumberInput input::placeholder {
    -webkit-text-fill-color: rgba(232,245,236,0.3) !important;
}
/* step buttons */
.stNumberInput button {
    color: #4ade80 !important;
    border-color: rgba(255,255,255,0.08) !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(34,197,94,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(34,197,94,0.4) !important;
    background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result card ── */
.result-outer {
    border-radius: 24px;
    padding: 3px;
    margin-top: 2rem;
    background: linear-gradient(135deg, rgba(74,222,128,0.4), rgba(16,110,48,0.2), rgba(74,222,128,0.1));
}
.result-inner {
    background: #060f08;
    border-radius: 22px;
    padding: 2.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-inner::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 200px;
    border-radius: 50%;
    pointer-events: none;
}
.result-eyebrow {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(232,245,236,0.4);
    margin-bottom: 0.6rem;
}
.result-tier {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1;
    margin: 0 0 0.4rem;
    letter-spacing: -0.02em;
}
.result-subtitle {
    color: rgba(232,245,236,0.35);
    font-size: 0.82rem;
    font-weight: 300;
}

/* ── Probability section ── */
.prob-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.6rem;
    margin-top: 1.2rem;
}
.prob-header {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(232,245,236,0.4);
    margin-bottom: 1.4rem;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 18px;
}
.prob-row:last-child { margin-bottom: 0; }
.prob-label {
    width: 76px;
    font-size: 0.78rem;
    font-weight: 500;
    color: rgba(232,245,236,0.5);
    flex-shrink: 0;
    text-align: right;
}
.prob-track {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 100px;
}
.prob-val {
    width: 44px;
    font-size: 0.78rem;
    font-weight: 600;
    text-align: right;
    flex-shrink: 0;
}

/* ── Tooltip hints ── */
.input-hint {
    font-size: 0.68rem;
    color: rgba(232,245,236,0.3);
    margin-top: -0.4rem;
    margin-bottom: 0.8rem;
    padding-left: 2px;
}

/* ── Divider ── */
.fancy-divider {
    position: relative;
    text-align: center;
    margin: 2.5rem 0;
}
.fancy-divider::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(74,222,128,0.15), transparent);
}
.fancy-divider span {
    position: relative;
    background: #030d07;
    padding: 0 12px;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(232,245,236,0.2);
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 0.5rem;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(232,245,236,0.15);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
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

    century_rate = st.number_input("Century Rate", min_value=0.0, value=0.0, step=0.1, format="%.1f",
                                   help="Centuries per 100 innings")
    st.markdown('<div class="input-hint">Per 100 innings</div>', unsafe_allow_html=True)

    duck_rate = st.number_input("Duck Rate", min_value=0.0, value=3.0, step=0.1, format="%.1f",
                                help="Ducks per 100 innings")
    st.markdown('<div class="input-hint">Per 100 innings</div>', unsafe_allow_html=True)

    Mat = st.number_input("Total Matches", min_value=1, value=50, step=1)
    st.markdown('<div class="input-hint">Career match count</div>', unsafe_allow_html=True)

with col2:
    HS = st.number_input("Highest Score", min_value=0, value=90, step=1)
    st.markdown('<div class="input-hint">Best innings score</div>', unsafe_allow_html=True)

    fifty_rate = st.number_input("Fifty Rate", min_value=0.0, value=5.0, step=0.1, format="%.1f",
                                 help="Fifties per 100 innings")
    st.markdown('<div class="input-hint">Per 100 innings</div>', unsafe_allow_html=True)

    no_rate = st.number_input("Not-Out %", min_value=0.0, max_value=100.0, value=10.0,
                              step=0.5, format="%.1f")
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
        st.error("Model files not found. Place cricket_model.pkl and cricket_scaler.pkl in the app directory.")
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

        # ── Probability bars ──
        prob_rows = ""
        for cls, prob in zip(model_ml.classes_, pred_proba):
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
<div class="footer">Cricket Performance Predictor &nbsp;·&nbsp; ML Model v1 &nbsp;·&nbsp; Built with Streamlit</div>
""", unsafe_allow_html=True)
