import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Churn Predictor",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────
# PREMIUM CSS — Beautiful in both light & dark
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── CSS Variables ── */
    :root {
        --indigo:       #6366F1;
        --indigo-dark:  #4338CA;
        --indigo-glow:  rgba(99,102,241,0.18);
        --teal:         #14B8A6;
        --rose:         #F43F5E;
        --amber:        #F59E0B;
        --emerald:      #10B981;
        --card-border:  rgba(99,102,241,0.2);
        --radius:       14px;
        --radius-sm:    8px;
        --shadow:       0 4px 24px rgba(99,102,241,0.13);
        --shadow-hover: 0 8px 32px rgba(99,102,241,0.25);
    }

    /* ── Page ── */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 2.5rem;
        max-width: 1220px;
    }

    /* ══════════════════════════════════════
       HEADER — gradient glass card
    ══════════════════════════════════════ */
    .app-header {
        position: relative;
        padding: 28px 32px;
        margin-bottom: 28px;
        border-radius: var(--radius);
        background: linear-gradient(135deg,
            rgba(99,102,241,0.12) 0%,
            rgba(20,184,166,0.08) 50%,
            rgba(99,102,241,0.05) 100%);
        border: 1px solid var(--card-border);
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .app-header::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 40px;
        width: 140px; height: 140px;
        background: radial-gradient(circle, rgba(20,184,166,0.13) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .app-header-badge {
        display: inline-block;
        background: linear-gradient(90deg, var(--indigo), var(--teal));
        color: #fff;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        padding: 3px 12px;
        border-radius: 99px;
        margin-bottom: 10px;
    }
    .app-header h1 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
        background: linear-gradient(135deg, var(--indigo) 0%, var(--teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .app-header p {
        margin: 0;
        font-size: 0.88rem;
        font-weight: 400;
        opacity: 0.6;
    }

    /* ══════════════════════════════════════
       SECTION LABEL
    ══════════════════════════════════════ */
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        background: linear-gradient(90deg, var(--indigo), var(--teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2px;
    }

    /* ══════════════════════════════════════
       METRIC CARDS
    ══════════════════════════════════════ */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg,
            rgba(99,102,241,0.07) 0%,
            rgba(99,102,241,0.02) 100%) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: var(--radius) !important;
        padding: 18px 20px !important;
        transition: all 0.25s ease !important;
        box-shadow: var(--shadow) !important;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: var(--shadow-hover) !important;
        border-color: var(--indigo) !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, var(--indigo), var(--teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
        opacity: 0.65;
        text-transform: uppercase;
    }

    /* ══════════════════════════════════════
       BUTTONS
    ══════════════════════════════════════ */
    .stButton > button {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border-radius: var(--radius-sm) !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--indigo) 0%, var(--indigo-dark) 100%) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
        padding: 0.55rem 1.4rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #7C7FF5 0%, var(--indigo) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0px) !important;
    }
    .stButton > button:not([kind="primary"]) {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--card-border) !important;
        background: rgba(99,102,241,0.05) !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--indigo) !important;
        background: rgba(99,102,241,0.1) !important;
        color: var(--indigo) !important;
        transform: translateY(-1px) !important;
    }

    /* ══════════════════════════════════════
       PROGRESS BAR
    ══════════════════════════════════════ */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--indigo) 0%, var(--teal) 100%) !important;
        border-radius: 99px !important;
    }
    .stProgress > div > div {
        border-radius: 99px !important;
        height: 10px !important;
        background: rgba(99,102,241,0.1) !important;
    }

    /* ══════════════════════════════════════
       INPUTS & SELECTS
    ══════════════════════════════════════ */
    .stTextInput > div > div > input {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--card-border) !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--indigo) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    }
    .stSelectbox > div > div {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--card-border) !important;
    }
    .stSlider [data-baseweb="slider"] {
        padding: 6px 0 !important;
    }

    /* ══════════════════════════════════════
       ALERT / INFO BOXES
    ══════════════════════════════════════ */
    .stSuccess {
        border-radius: var(--radius-sm) !important;
        border-left: 4px solid var(--emerald) !important;
        background: rgba(16,185,129,0.08) !important;
    }
    .stInfo {
        border-radius: var(--radius-sm) !important;
        border-left: 4px solid var(--indigo) !important;
        background: rgba(99,102,241,0.07) !important;
    }
    .stError {
        border-radius: var(--radius-sm) !important;
        border-left: 4px solid var(--rose) !important;
        background: rgba(244,63,94,0.07) !important;
    }
    .stWarning {
        border-radius: var(--radius-sm) !important;
        border-left: 4px solid var(--amber) !important;
        background: rgba(245,158,11,0.07) !important;
    }

    /* ══════════════════════════════════════
       DATAFRAME
    ══════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--card-border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow) !important;
    }

    /* ══════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════ */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--card-border) !important;
        background: linear-gradient(180deg,
            rgba(99,102,241,0.04) 0%,
            rgba(99,102,241,0.01) 100%) !important;
    }
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        background: linear-gradient(90deg, var(--indigo), var(--teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        border-radius: var(--radius-sm) !important;
        padding: 6px 10px !important;
        transition: all 0.15s !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(99,102,241,0.08) !important;
    }
    [data-testid="stSidebar"] .stInfo {
        border-radius: var(--radius-sm) !important;
        font-size: 0.82rem !important;
    }

    /* ══════════════════════════════════════
       DIVIDERS
    ══════════════════════════════════════ */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg,
            transparent,
            rgba(99,102,241,0.3),
            rgba(20,184,166,0.3),
            transparent) !important;
        margin: 24px 0 !important;
    }

    /* ══════════════════════════════════════
       HEADINGS
    ══════════════════════════════════════ */
    h2 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
    }
    h3 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.2px !important;
    }

    /* ══════════════════════════════════════
       DOWNLOAD BUTTON
    ══════════════════════════════════════ */
    [data-testid="stDownloadButton"] > button {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--card-border) !important;
        background: rgba(99,102,241,0.06) !important;
        transition: all 0.2s !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        border-color: var(--indigo) !important;
        background: rgba(99,102,241,0.12) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.2) !important;
    }

    /* ══════════════════════════════════════
       MULTISELECT TAGS
    ══════════════════════════════════════ */
    [data-baseweb="tag"] {
        background: linear-gradient(135deg, var(--indigo), var(--indigo-dark)) !important;
        border-radius: 99px !important;
        color: #fff !important;
    }

    /* ══════════════════════════════════════
       SPINNER
    ══════════════════════════════════════ */
    .stSpinner > div > div {
        border-top-color: var(--indigo) !important;
    }

    /* ══════════════════════════════════════
       CAPTION / SMALL TEXT
    ══════════════════════════════════════ */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 0.8rem !important;
        opacity: 0.6 !important;
        font-style: italic;
    }

    /* ══════════════════════════════════════
       RADIO SELECTED STATE
    ══════════════════════════════════════ */
    [data-testid="stSidebar"] [data-baseweb="radio"] svg {
        fill: var(--indigo) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MATPLOTLIB — adaptive to both modes
# Reads Streamlit's background dynamically
# ─────────────────────────────────────────────
def apply_chart_style():
    """Returns fig, ax with transparent background and adaptive text colours."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_edgecolor("rgba(99,102,241,0.2)")
    ax.tick_params(colors="#9CA3AF")
    ax.xaxis.label.set_color("#9CA3AF")
    ax.yaxis.label.set_color("#9CA3AF")
    ax.title.set_color("#6366F1")
    ax.title.set_fontsize(11)
    ax.title.set_fontweight('bold')
    return fig, ax


# ─────────────────────────────────────────────
# GROQ CLIENT  — reads from Streamlit secrets
# In Streamlit Cloud → Settings → Secrets:
#   GROQ_API_KEY = "gsk_..."
# ─────────────────────────────────────────────
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    client = None

GROQ_MODEL = "llama-3.3-70b-versatile"

def ask_groq(prompt: str) -> str:
    """Call Groq and return text, or a graceful error string."""
    if client is None:
        return "⚠️ AI unavailable — GROQ_API_KEY not set in Streamlit secrets."
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI request failed: {e}"


# ─────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, 'models', 'churn_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, 'Data', 'customer_features.csv'))

model, scaler = load_model()
df = load_data()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
    <div class='app-header-badge'>✦ Retention Intelligence System</div>
    <h1>🤖 AI Churn Predictor</h1>
    <p>Predict customer churn with Machine Learning &amp; explain results with Groq AI in real-time</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Navigation")
    page = st.radio(
        "Go to:",
        ["📊 Dashboard", "🔍 Churn Predictor", "🤖 AI Assistant", "📈 Analytics"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    st.markdown("**Project Info**")
    st.info("Built with Python, Scikit-learn, Groq AI & Streamlit")

    churn_rate = round(df['Churned'].mean() * 100, 2)
    color = "#F43F5E" if churn_rate > 25 else "#10B981"
    st.markdown(f"**Total Customers:** `{df.shape[0]}`")
    st.markdown(
        f"**Churn Rate:** <span style='color:{color};font-weight:700'>{churn_rate}%</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"**AI Model:** `{GROQ_MODEL}`")


# ─────────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ─────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown("<div class='section-label'>Overview</div>", unsafe_allow_html=True)
    st.subheader("Business Overview Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{df.shape[0]:,}")
    with col2:
        churned = int(df['Churned'].sum())
        st.metric("Churned Customers", f"{churned:,}", delta=f"-{churned}", delta_color="inverse")
    with col3:
        st.metric("Churn Rate", f"{churn_rate}%")
    with col4:
        avg_rev = round(df['Monetary'].mean(), 2)
        st.metric("Avg Revenue / Customer", f"${avg_rev:,.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = apply_chart_style()
        counts = df['Churned'].value_counts()
        wedge_props = {'linewidth': 2, 'edgecolor': 'none'}
        ax.pie(
            counts,
            labels=['Retained', 'Churned'],
            autopct='%1.1f%%',
            colors=['#6366F1', '#F43F5E'],
            textprops={'color': '#9CA3AF', 'fontsize': 11},
            wedgeprops=wedge_props,
            startangle=90,
            pctdistance=0.75
        )
        ax.set_ylabel('')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn Rate by Category")
        fig, ax = apply_chart_style()
        churn_cat = df.groupby('Fav_Category')['Churned'].mean() * 100
        bars = churn_cat.plot(kind='bar', ax=ax, color='#6366F1',
                               edgecolor='none', width=0.55)
        ax.set_xlabel('')
        ax.set_ylabel('Churn Rate (%)', color='#9CA3AF')
        ax.set_title('Churn Rate by Category (%)', color='#6366F1')
        plt.xticks(rotation=0)
        # label each bar
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f'{bar.get_height():.1f}%',
                ha='center', va='bottom', color='#9CA3AF', fontsize=9
            )
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recency vs Churn")
        fig, ax = apply_chart_style()
        data_not = df[df['Churned'] == 0]['Recency']
        data_yes = df[df['Churned'] == 1]['Recency']
        bp = ax.boxplot(
            [data_not, data_yes],
            labels=['Not Churned', 'Churned'],
            patch_artist=True,
            boxprops=dict(facecolor='#6366F1', alpha=0.45, linewidth=0),
            medianprops=dict(color='#F43F5E', linewidth=2.5),
            whiskerprops=dict(color='#9CA3AF'),
            capprops=dict(color='#9CA3AF'),
            flierprops=dict(markerfacecolor='#6366F1', alpha=0.4, markersize=4)
        )
        ax.set_title('Recency Distribution by Churn', color='#6366F1')
        ax.set_ylabel('Days Since Last Purchase', color='#9CA3AF')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Revenue Distribution")
        fig, ax = apply_chart_style()
        ax.hist(df['Monetary'], bins=30, color='#6366F1', alpha=0.8,
                edgecolor='none', linewidth=0)
        ax.set_title('Customer Revenue Distribution', color='#6366F1')
        ax.set_xlabel('Revenue ($)', color='#9CA3AF')
        ax.set_ylabel('Count', color='#9CA3AF')
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────
# PAGE 2 — CHURN PREDICTOR
# ─────────────────────────────────────────────
elif page == "🔍 Churn Predictor":
    st.markdown("<div class='section-label'>Prediction</div>", unsafe_allow_html=True)
    st.subheader("Customer Churn Predictor")
    st.caption("Enter customer details below to predict churn risk and get an AI explanation.")

    col1, col2 = st.columns(2)
    with col1:
        recency   = st.slider("📅 Days Since Last Purchase", 0, 365, 90)
        frequency = st.slider("🔁 Total Orders", 1, 20, 5)
        monetary  = st.slider("💰 Total Revenue ($)", 0, 10000, 2000, step=50)
    with col2:
        avg_order = st.slider("🧾 Avg Order Value ($)", 0, 5000, 500, step=25)
        total_qty = st.slider("📦 Total Quantity Purchased", 1, 50, 10)
        category  = st.selectbox("🛍️ Favourite Category", ['Electronics', 'Accessories', 'Office'])
        region    = st.selectbox("🌍 Region", ['North', 'South', 'East', 'West'])

    st.markdown("")
    predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

    if predict_btn:
        cat_map = {'Accessories': 0, 'Electronics': 1, 'Office': 2}
        reg_map = {'East': 0, 'North': 1, 'South': 2, 'West': 3}

        features = np.array([[
            recency, frequency, monetary,
            avg_order, total_qty,
            cat_map[category], reg_map[region]
        ]])
        features_scaled = scaler.transform(features)
        prediction  = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        pct = round(probability * 100, 1)

        st.markdown("---")

        res_col1, res_col2 = st.columns([1.5, 1])
        with res_col1:
            if prediction == 1:
                st.error(f"⚠️ **HIGH CHURN RISK** — {pct}% probability")
            else:
                st.success(f"✅ **LOW CHURN RISK** — {pct}% probability")

            st.caption("Churn probability score:")
            st.progress(int(probability * 100))

        with res_col2:
            # Mini gauge using matplotlib
            fig, ax = plt.subplots(figsize=(3, 1.6))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            ax.barh(0, 100, color='#E0E0E0', height=0.4)
            bar_color = '#F43F5E' if prediction == 1 else '#10B981'
            ax.barh(0, pct, color=bar_color, height=0.4)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.tick_params(colors='#9CA3AF', labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"Score: {pct}%", color=bar_color, fontsize=11, fontweight='bold', pad=6)
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.subheader("🤖 AI Explanation")

        with st.spinner("Analysing customer profile…"):
            prompt = f"""
Analyse this e-commerce customer and explain the churn risk in clear business language.

Customer Profile:
- Recency: {recency} days since last purchase
- Frequency: {frequency} total orders
- Total Revenue: ${monetary:,}
- Avg Order Value: ${avg_order:,}
- Preferred Category: {category}
- Region: {region}

Model Prediction: {'HIGH RISK — Likely to Churn' if prediction == 1 else 'LOW RISK — Active Customer'}
Churn Probability: {pct}%

Respond with exactly:
1. **Risk Assessment** — one paragraph
2. **Key Drivers** — 2-3 bullet points
3. **Recommended Actions** — 2 bullet points

Be concise, specific, and business-focused. No generic filler.
"""
            answer = ask_groq(prompt)

        st.info(answer)


# ─────────────────────────────────────────────
# PAGE 3 — AI ASSISTANT
# ─────────────────────────────────────────────
elif page == "🤖 AI Assistant":
    st.markdown("<div class='section-label'>AI Assistant</div>", unsafe_allow_html=True)
    st.subheader("AI Data Assistant")
    st.caption("Ask any business question about your customer data.")

    # Quick-fire buttons
    st.markdown("**Quick questions:**")
    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        if st.button("📊 Overall churn rate?", use_container_width=True):
            st.session_state.question = "What is the overall churn rate and is it concerning?"
    with qcol2:
        if st.button("⚠️ Who is most at risk?", use_container_width=True):
            st.session_state.question = "Which customer segments are most at risk of churning?"
    with qcol3:
        if st.button("💡 Retention strategy?", use_container_width=True):
            st.session_state.question = "What retention strategies would you recommend?"

    question = st.text_input(
        "Or type your own question:",
        value=st.session_state.get('question', ''),
        placeholder="e.g. Which region has the highest churn rate?"
    )

    if st.button("🔍 Ask AI", use_container_width=True, type="primary"):
        if question.strip():
            with st.spinner("Thinking…"):
                data_summary = f"""
Dataset Summary:
- Total customers:  {df.shape[0]}
- Churned:          {int(df['Churned'].sum())} ({churn_rate}%)
- Avg recency:      {round(df['Recency'].mean(), 1)} days
- Avg revenue:      ${round(df['Monetary'].mean(), 2):,.2f}
- Top category:     {df['Fav_Category'].mode()[0]}
- Top region:       {df['Region'].mode()[0]}
- High-risk count:  {int(df[df['Churned']==1].shape[0])}
- Low-risk count:   {int(df[df['Churned']==0].shape[0])}
"""
                prompt = f"""
You are a senior data analyst for an e-commerce company.
Answer the following business question using the data summary below.
Give a clear, actionable answer in 3-4 sentences. No generic filler.

{data_summary}

Question: {question}
"""
                answer = ask_groq(prompt)

            st.success("**AI Answer:**")
            st.write(answer)
        else:
            st.warning("Please enter a question first.")

    st.markdown("---")

    # Full report section
    st.markdown("<div class='section-label'>Report Generator</div>", unsafe_allow_html=True)
    st.subheader("Generate AI Business Report")

    if st.button("📊 Generate Full Report", use_container_width=True):
        with st.spinner("Writing professional report…"):
            prompt = f"""
Generate a structured professional churn analysis report for an e-commerce business.

Data:
- Total Customers:  {df.shape[0]}
- Churned:          {int(df['Churned'].sum())} ({churn_rate}%)
- Retained:         {int((df['Churned']==0).sum())}
- Avg Revenue:      ${round(df['Monetary'].mean(), 2):,.2f}
- Top Category:     {df['Fav_Category'].mode()[0]}
- Top Region:       {df['Region'].mode()[0]}

Sections required:
1. Executive Summary
2. Key Findings (3 bullet points)
3. Risk Assessment
4. Actionable Recommendations (3 bullet points)
5. Conclusion

Use markdown formatting. Be concise, data-driven, and professional.
"""
            report = ask_groq(prompt)

        st.markdown(report)
        st.download_button(
            label="📥 Download Report (.txt)",
            data=report,
            file_name="AI_Churn_Report.txt",
            mime="text/plain",
            use_container_width=True
        )


# ─────────────────────────────────────────────
# PAGE 4 — ANALYTICS
# ─────────────────────────────────────────────
elif page == "📈 Analytics":
    st.markdown("<div class='section-label'>Analytics</div>", unsafe_allow_html=True)
    st.subheader("Customer Analytics")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_region = st.multiselect(
            "Filter by Region:",
            options=df['Region'].unique(),
            default=list(df['Region'].unique())
        )
    with filter_col2:
        selected_category = st.multiselect(
            "Filter by Category:",
            options=df['Fav_Category'].unique(),
            default=list(df['Fav_Category'].unique())
        )

    filtered_df = df[
        df['Region'].isin(selected_region) &
        df['Fav_Category'].isin(selected_category)
    ]

    if filtered_df.empty:
        st.warning("No data matches your filters. Please adjust the selections above.")
        st.stop()

    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    filt_churn = round(filtered_df['Churned'].mean() * 100, 2)
    with m1:
        st.metric("Filtered Customers", f"{filtered_df.shape[0]:,}")
    with m2:
        st.metric("Churned", f"{int(filtered_df['Churned'].sum()):,}")
    with m3:
        delta_color = "inverse" if filt_churn > churn_rate else "normal"
        st.metric("Churn Rate", f"{filt_churn}%",
                  delta=f"{round(filt_churn - churn_rate, 2)}% vs total",
                  delta_color=delta_color)
    with m4:
        st.metric("Avg Revenue", f"${round(filtered_df['Monetary'].mean(), 2):,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Rate by Region")
        fig, ax = apply_chart_style()
        churn_region = filtered_df.groupby('Region')['Churned'].mean() * 100
        colors_region = ['#F43F5E' if v == churn_region.max() else '#6366F1'
                         for v in churn_region]
        churn_region.plot(kind='bar', ax=ax, color=colors_region,
                          edgecolor='none', width=0.55)
        ax.set_ylabel('Churn Rate (%)', color='#9CA3AF')
        ax.set_title('Churn Rate by Region (%)', color='#6366F1')
        plt.xticks(rotation=0)
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%',
                ha='center', va='bottom', color='#9CA3AF', fontsize=9
            )
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Avg Revenue by Region")
        fig, ax = apply_chart_style()
        rev_region = filtered_df.groupby('Region')['Monetary'].mean()
        colors_rev = ['#10B981' if v == rev_region.max() else '#6366F1'
                      for v in rev_region]
        rev_region.plot(kind='bar', ax=ax, color=colors_rev,
                        edgecolor='none', width=0.55)
        ax.set_ylabel('Avg Revenue ($)', color='#9CA3AF')
        ax.set_title('Avg Revenue by Region', color='#6366F1')
        plt.xticks(rotation=0)
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'${bar.get_height():,.0f}',
                ha='center', va='bottom', color='#9CA3AF', fontsize=9
            )
        st.pyplot(fig)
        plt.close()

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    num_cols = ['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value',
                'Total_Quantity', 'Churned']
    available = [c for c in num_cols if c in filtered_df.columns]
    if len(available) >= 3:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        corr = filtered_df[available].corr()
        sns.heatmap(
            corr, annot=True, fmt='.2f', ax=ax,
            cmap='RdBu_r', center=0,
            linewidths=0.5, linecolor='none',
            annot_kws={"size": 9, "color": "white"},
            cbar_kws={"shrink": 0.8}
        )
        ax.tick_params(colors='#9CA3AF', labelsize=9)
        ax.set_title('Feature Correlations', color='#6366F1', pad=10)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("📋 Customer Data Table")
    display_df = filtered_df.sort_values('Monetary', ascending=False).reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, height=380)

    st.download_button(
        label="📥 Download Filtered Data (.csv)",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_customers.csv",
        mime="text/csv",
        use_container_width=True
    )