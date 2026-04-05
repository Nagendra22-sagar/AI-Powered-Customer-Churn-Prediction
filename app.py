import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# PAGE CONFIG
st.set_page_config(
    page_title="AI Churn Predictor",
    page_icon="🤖",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric {
        background-color: #1E3A5F;
        border: 1px solid #00B4D8;
        border-radius: 10px;
        padding: 10px;
    }
    h1, h2, h3 { color: #00B4D8; }
</style>
""", unsafe_allow_html=True)

# GROQ CLIENT
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# BASE DIRECTORY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LOAD MODEL & DATA
@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, 'models', 'churn_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'Data', 'customer_features.csv'))
    return df

model, scaler = load_model()
df = load_data()

# HEADER
st.markdown("""
    <h1 style='text-align: center; color: #00B4D8;'>
    🤖 AI-Powered Customer Churn Prediction
    </h1>
    <h4 style='text-align: center; color: white;'>
    Retention Intelligence System
    </h4>
""", unsafe_allow_html=True)

st.markdown("---")

# SIDEBAR
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["📊 Dashboard", "🔍 Churn Predictor", "🤖 AI Assistant", "📈 Analytics"]
)

# PAGE 1 — DASHBOARD
if page == "📊 Dashboard":
    st.header("📊 Business Overview Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", df.shape[0])
    with col2:
        st.metric("Churned Customers", df['Churned'].sum())
    with col3:
        st.metric("Churn Rate", f"{round(df['Churned'].mean()*100, 2)}%")
    with col4:
        st.metric("Avg Revenue/Customer", f"${round(df['Monetary'].mean(), 2)}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        df['Churned'].value_counts().plot(
            kind='pie',
            labels=['Retained', 'Churned'],
            autopct='%1.1f%%',
            colors=['#00B4D8', '#FF4444'],
            ax=ax
        )
        ax.set_ylabel('')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Churn by Category")
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_cat = df.groupby('Fav_Category')['Churned'].mean() * 100
        churn_cat.plot(kind='bar', color='#00B4D8', ax=ax)
        ax.set_title('Churn Rate by Category (%)')
        ax.set_xlabel('')
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recency vs Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        churned_0 = df[df['Churned']==0]['Recency']
        churned_1 = df[df['Churned']==1]['Recency']
        ax.boxplot([churned_0, churned_1],
                   labels=['Not Churned', 'Churned'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#00B4D8', alpha=0.6),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title('Recency vs Churn')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Revenue Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        df['Monetary'].hist(bins=30, color='#00B4D8', ax=ax)
        ax.set_title('Customer Revenue Distribution')
        st.pyplot(fig)

# PAGE 2 — CHURN PREDICTOR
elif page == "🔍 Churn Predictor":
    st.header("🔍 Customer Churn Predictor")
    st.write("Enter customer details to predict churn risk!")
    
    col1, col2 = st.columns(2)
    with col1:
        recency = st.slider("Days Since Last Purchase", 0, 365, 90)
        frequency = st.slider("Total Orders", 1, 20, 5)
        monetary = st.slider("Total Revenue ($)", 0, 10000, 2000)
    
    with col2:
        avg_order = st.slider("Avg Order Value ($)", 0, 5000, 500)
        total_qty = st.slider("Total Quantity", 1, 50, 10)
        category = st.selectbox("Favourite Category", ['Electronics', 'Accessories', 'Office'])
        region = st.selectbox("Region", ['North', 'South', 'East', 'West'])
    
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        cat_map = {'Accessories': 0, 'Electronics': 1, 'Office': 2}
        reg_map = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
        
        features = np.array([[
            recency, frequency, monetary,
            avg_order, total_qty,
            cat_map[category], reg_map[region]
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        st.markdown("---")
        
        if prediction == 1:
            st.error(f"⚠️ HIGH CHURN RISK — {round(probability*100, 1)}% probability")
        else:
            st.success(f"✅ LOW CHURN RISK — {round(probability*100, 1)}% probability")
        
        st.subheader("🤖 AI Explanation")
        with st.spinner("Getting AI analysis..."):
            prompt = f"""
            Analyze this customer and explain churn risk:
            - Recency: {recency} days
            - Frequency: {frequency} orders
            - Revenue: ${monetary}
            - Avg Order: ${avg_order}
            - Category: {category}
            - Region: {region}
            - Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}
            - Probability: {round(probability*100, 1)}%
            Give 3-4 lines business explanation and recommendations.
            """
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            st.info(response.choices[0].message.content)

# PAGE 3 — AI ASSISTANT
elif page == "🤖 AI Assistant":
    st.header("🤖 AI Data Assistant")
    st.write("Ask any business question about your customer data!")
    
    st.subheader("Quick Questions:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What is the churn rate?"):
            st.session_state.question = "What is the overall churn rate and is it concerning?"
    with col2:
        if st.button("Who are at risk?"):
            st.session_state.question = "Which customers are most at risk of churning?"
    with col3:
        if st.button("Recommendations?"):
            st.session_state.question = "What strategy would you recommend to improve retention?"
    
    question = st.text_input(
        "Or ask your own question:",
        value=st.session_state.get('question', ''),
        placeholder="e.g. Which region has highest churn rate?"
    )
    
    if st.button("🔍 Ask AI", use_container_width=True):
        if question:
            with st.spinner("AI is thinking..."):
                summary = f"""
                Dataset: {df.shape[0]} customers
                Churned: {df['Churned'].sum()} ({round(df['Churned'].mean()*100,2)}%)
                Avg Recency: {round(df['Recency'].mean(),2)} days
                Avg Revenue: ${round(df['Monetary'].mean(),2)}
                Top Category: {df['Fav_Category'].mode()[0]}
                Top Region: {df['Region'].mode()[0]}
                """
                prompt = f"""
                You are a data analyst. Answer based on this data:
                {summary}
                Question: {question}
                Give clear business answer in 3-4 lines.
                """
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.success("**AI Answer:**")
                st.write(response.choices[0].message.content)
    
    st.markdown("---")
    st.subheader("📄 Generate AI Business Report")
    
    if st.button("📊 Generate Full Report", use_container_width=True):
        with st.spinner("Generating report..."):
            prompt = f"""
            Generate a professional churn analysis report:
            - Total Customers: {df.shape[0]}
            - Churned: {df['Churned'].sum()} ({round(df['Churned'].mean()*100,2)}%)
            - Retained: {(df['Churned']==0).sum()}
            - Avg Revenue: ${round(df['Monetary'].mean(),2)}
            - Top Category: {df['Fav_Category'].mode()[0]}
            - Top Region: {df['Region'].mode()[0]}
            Include: 1. Executive Summary 2. Key Findings 3. Risk Assessment 4. Recommendations
            """
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            report = response.choices[0].message.content
            st.write(report)
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name="AI_Churn_Report.txt",
                mime="text/plain"
            )

# PAGE 4 — ANALYTICS
elif page == "📈 Analytics":
    st.header("📈 Customer Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_region = st.multiselect(
            "Filter by Region:",
            options=df['Region'].unique(),
            default=df['Region'].unique()
        )
    with col2:
        selected_category = st.multiselect(
            "Filter by Category:",
            options=df['Fav_Category'].unique(),
            default=df['Fav_Category'].unique()
        )
    
    filtered_df = df[
        (df['Region'].isin(selected_region)) &
        (df['Fav_Category'].isin(selected_category))
    ]
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filtered Customers", filtered_df.shape[0])
    with col2:
        st.metric("Churn Rate", f"{round(filtered_df['Churned'].mean()*100, 2)}%")
    with col3:
        st.metric("Avg Revenue", f"${round(filtered_df['Monetary'].mean(), 2)}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn by Region")
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_region = filtered_df.groupby('Region')['Churned'].mean() * 100
        churn_region.plot(kind='bar', color='#00B4D8', ax=ax)
        ax.set_title('Churn Rate by Region (%)')
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Revenue by Region")
        fig, ax = plt.subplots(figsize=(6, 4))
        rev_region = filtered_df.groupby('Region')['Monetary'].mean()
        rev_region.plot(kind='bar', color='#0077B6', ax=ax)
        ax.set_title('Avg Revenue by Region')
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    st.subheader("📋 Customer Data Table")
    st.dataframe(
        filtered_df.sort_values('Monetary', ascending=False),
        use_container_width=True
    )
    
    st.download_button(
        label="📥 Download Filtered Data",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_customers.csv",
        mime="text/csv"
    )