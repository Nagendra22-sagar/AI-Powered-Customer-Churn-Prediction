import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from groq import Groq

st.set_page_config(page_title="AI Churn Predictor", page_icon="🤖", layout="wide")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
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

st.title("🤖 AI-Powered Customer Churn Prediction")
st.caption("Retention Intelligence System")
st.divider()

st.sidebar.title("🎛️ Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dashboard", "🔍 Churn Predictor", "🤖 AI Assistant", "📈 Analytics"])
st.sidebar.divider()
st.sidebar.markdown("**Project Info**")
st.sidebar.info("Built with Python, Scikit-learn, Groq AI & Streamlit")
st.sidebar.metric("Total Customers", df.shape[0])
st.sidebar.metric("Churn Rate", f"{round(df['Churned'].mean()*100, 2)}%")

if page == "📊 Dashboard":
    st.subheader("📊 Business Overview Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("👥 Total Customers", df.shape[0])
    with col2: st.metric("⚠️ Churned", df['Churned'].sum())
    with col3: st.metric("📉 Churn Rate", f"{round(df['Churned'].mean()*100, 2)}%")
    with col4: st.metric("💰 Avg Revenue", f"${round(df['Monetary'].mean(), 2)}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        churn_data = df['Churned'].value_counts().rename({0: 'Retained', 1: 'Churned'})
        st.bar_chart(churn_data)
    with col2:
        st.subheader("Churn Rate by Category (%)")
        churn_cat = (df.groupby('Fav_Category')['Churned'].mean() * 100).round(2)
        st.bar_chart(churn_cat)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recency Stats by Churn Status")
        retained = df[df['Churned']==0]['Recency'].describe().round(2)
        churned = df[df['Churned']==1]['Recency'].describe().round(2)
        st.dataframe(pd.DataFrame({'Retained': retained, 'Churned': churned}), use_container_width=True)
    with col2:
        st.subheader("Avg Revenue by Category")
        rev_cat = df.groupby('Fav_Category')['Monetary'].mean().round(2)
        st.bar_chart(rev_cat)

elif page == "🔍 Churn Predictor":
    st.subheader("🔍 Customer Churn Predictor")
    st.write("Enter customer details to predict churn risk!")

    col1, col2 = st.columns(2)
    with col1:
        recency = st.slider("📅 Days Since Last Purchase", 0, 365, 90)
        frequency = st.slider("🛒 Total Orders", 1, 20, 5)
        monetary = st.slider("💵 Total Revenue ($)", 0, 10000, 2000)
    with col2:
        avg_order = st.slider("📦 Avg Order Value ($)", 0, 5000, 500)
        total_qty = st.slider("📊 Total Quantity", 1, 50, 10)
        category = st.selectbox("🏷️ Favourite Category", ['Electronics', 'Accessories', 'Office'])
        region = st.selectbox("📍 Region", ['North', 'South', 'East', 'West'])

    if st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary"):
        cat_map = {'Accessories': 0, 'Electronics': 1, 'Office': 2}
        reg_map = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
        features = np.array([[recency, frequency, monetary, avg_order, total_qty, cat_map[category], reg_map[region]]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1: st.error("⚠️ HIGH CHURN RISK")
            else: st.success("✅ LOW CHURN RISK")
        with col2:
            st.metric("Churn Probability", f"{round(probability*100, 1)}%")
        st.progress(int(probability * 100))
        st.subheader("🤖 AI Explanation")
        with st.spinner("Getting AI analysis..."):
            prompt = f"""Analyze this customer churn risk:
            - Recency: {recency} days, Frequency: {frequency} orders, Revenue: ${monetary}
            - Avg Order: ${avg_order}, Category: {category}, Region: {region}
            - Prediction: {'High Risk' if prediction == 1 else 'Low Risk'} ({round(probability*100, 1)}%)
            Give: 1) Risk Assessment 2) Key reasons 3) Recommendations. Be concise."""
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            st.info(response.choices[0].message.content)

elif page == "🤖 AI Assistant":
    st.subheader("🤖 AI Data Assistant")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Churn Rate?", use_container_width=True):
            st.session_state.question = "What is the overall churn rate and is it concerning?"
    with col2:
        if st.button("⚠️ Who are at risk?", use_container_width=True):
            st.session_state.question = "Which customers are most at risk of churning?"
    with col3:
        if st.button("💡 Recommendations?", use_container_width=True):
            st.session_state.question = "What strategy would you recommend to improve retention?"

    question = st.text_input("Or ask your own question:", value=st.session_state.get('question', ''), placeholder="e.g. Which region has highest churn rate?")
    if st.button("🔍 Ask AI", use_container_width=True, type="primary"):
        if question:
            with st.spinner("AI is thinking..."):
                summary = f"Customers: {df.shape[0]}, Churned: {df['Churned'].sum()} ({round(df['Churned'].mean()*100,2)}%), Avg Recency: {round(df['Recency'].mean(),2)} days, Avg Revenue: ${round(df['Monetary'].mean(),2)}, Top Category: {df['Fav_Category'].mode()[0]}, Top Region: {df['Region'].mode()[0]}"
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": f"Data analyst. Answer based on: {summary}\nQuestion: {question}\nGive clear answer in 3-4 lines."}]
                )
                st.success("**AI Answer:**")
                st.write(response.choices[0].message.content)

    st.divider()
    st.subheader("📄 Generate AI Business Report")
    if st.button("📊 Generate Full Report", use_container_width=True):
        with st.spinner("Generating report..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"Professional churn report: {df.shape[0]} customers, {df['Churned'].sum()} churned ({round(df['Churned'].mean()*100,2)}%), Avg Revenue ${round(df['Monetary'].mean(),2)}. Include: 1.Executive Summary 2.Key Findings 3.Risk Assessment 4.Recommendations"}]
            )
            report = response.choices[0].message.content
            st.write(report)
            st.download_button("📥 Download Report", data=report, file_name="AI_Churn_Report.txt", mime="text/plain")

elif page == "📈 Analytics":
    st.subheader("📈 Customer Analytics")
    col1, col2 = st.columns(2)
    with col1:
        selected_region = st.multiselect("Filter by Region:", options=df['Region'].unique(), default=df['Region'].unique())
    with col2:
        selected_category = st.multiselect("Filter by Category:", options=df['Fav_Category'].unique(), default=df['Fav_Category'].unique())

    filtered_df = df[(df['Region'].isin(selected_region)) & (df['Fav_Category'].isin(selected_category))]
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("👥 Customers", filtered_df.shape[0])
    with col2: st.metric("📉 Churn Rate", f"{round(filtered_df['Churned'].mean()*100, 2)}%")
    with col3: st.metric("💰 Avg Revenue", f"${round(filtered_df['Monetary'].mean(), 2)}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Rate by Region (%)")
        st.bar_chart((filtered_df.groupby('Region')['Churned'].mean() * 100).round(2))
    with col2:
        st.subheader("Avg Revenue by Region")
        st.bar_chart(filtered_df.groupby('Region')['Monetary'].mean().round(2))

    st.subheader("📋 Customer Data Table")
    st.dataframe(filtered_df.sort_values('Monetary', ascending=False), use_container_width=True)
    st.download_button("📥 Download Filtered Data", data=filtered_df.to_csv(index=False), file_name="filtered_customers.csv", mime="text/csv")