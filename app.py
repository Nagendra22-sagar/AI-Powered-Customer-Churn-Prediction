import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# DARK MODE FIX
plt.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'axes.edgecolor': 'white',
    'axes.titlecolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white'
})

# PAGE CONFIG
st.set_page_config(page_title="AI Churn Predictor", page_icon="🤖", layout="wide")

# CSS
st.markdown("""
<style>
.main .block-container {padding-top:1rem;padding-bottom:1rem;}
[data-testid="metric-container"] {border:1px solid rgba(49,51,63,0.2);border-radius:10px;padding:10px 15px;}
.main-header {text-align:center;padding:10px 0;border-bottom:2px solid #1F3864;margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

# GROQ CLIENT (SECURE)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# BASE PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LOAD MODEL
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

# HEADER
st.markdown("<h1 style='text-align:center;color:#1F3864;'>🤖 AI Customer Churn Prediction</h1>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predictor", "AI Assistant", "Analytics"])

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", df.shape[0])
    col2.metric("Churned", df['Churned'].sum())
    col3.metric("Churn Rate", f"{round(df['Churned'].mean()*100,2)}%")

    col1, col2 = st.columns(2)

    # PIE
    with col1:
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')

        df['Churned'].value_counts().plot(
            kind='pie',
            labels=['Retained','Churned'],
            autopct='%1.1f%%',
            colors=['#1F77B4','#FF4444'],
            textprops={'color':'white'},
            ax=ax
        )

        ax.set_ylabel('')
        ax.tick_params(colors='white')
        st.pyplot(fig)

    # BAR
    with col2:
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')

        df.groupby('Fav_Category')['Churned'].mean().plot(kind='bar', ax=ax, color='#1F3864')

        ax.set_title("Churn by Category", color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)

# ---------------- PREDICTOR ----------------
elif page == "Predictor":
    st.subheader("Churn Predictor")

    recency = st.slider("Recency", 0, 365, 90)
    freq = st.slider("Orders", 1, 20, 5)
    money = st.slider("Revenue", 0, 10000, 2000)

    if st.button("Predict"):
        features = np.array([[recency, freq, money, 500, 10, 1, 1]])
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        st.write("Prediction:", "Churn" if pred==1 else "No Churn")
        st.write("Probability:", round(prob*100,2), "%")

# ---------------- AI ----------------
elif page == "AI Assistant":
    st.subheader("AI Assistant")

    question = st.text_input("Ask something")

    if st.button("Ask"):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":question}]
        )
        st.write(response.choices[0].message.content)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":
    st.subheader("Analytics")

    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    df['Monetary'].hist(ax=ax, color='#1F3864')

    ax.set_title("Revenue Distribution", color='white')
    ax.tick_params(colors='white')

    st.pyplot(fig)