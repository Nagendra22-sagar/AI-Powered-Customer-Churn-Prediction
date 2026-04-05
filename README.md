# 🤖 AI-Powered Customer Churn Prediction & Retention Intelligence System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nagendra-churn-predictor.streamlit.app)

## 📌 Project Overview
An end-to-end AI-powered customer churn prediction system that combines Machine Learning, Generative AI, and Business Intelligence to identify at-risk customers and provide actionable retention strategies.

## 📸 Project Screenshots

### 🖥️ Power BI Dashboard — Business Overview
![Business Overview](dashboard_overview.png)

### 📊 Power BI Dashboard — Revenue Analysis
![Revenue Analysis](revenue_analysis.png)

### 🔴 Power BI Dashboard — Churn Analysis
![Churn Analysis](churn_analysis.png)

### 🗺️ Power BI Dashboard — Product & Region
![Product Region](product_region_analysis.png)

### 🌐 Streamlit Web App
![Streamlit App](streamlit_app.png)

## 🔗 Live Demo
👉 **[Click here to view Live App](https://nagendra-churn-predictor.streamlit.app)**

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Data Analysis & ML |
| SQL (MySQL) | Data Extraction & Querying |
| Excel | Data Summary & Pivot Tables |
| Power BI | Interactive Dashboard |
| Scikit-learn | Machine Learning Models |
| Groq LLM API | AI Explanations & Chatbot |
| Streamlit | Web Application |
| GitHub | Version Control |

## 📊 Dataset
- **594 unique customers**
- **3,000 transactions**
- **Full year data (Jan-Dec 2023)**
- **Zero missing values**
- **10 features:** Order_ID, Customer_ID, Order_Date, Product, Category, Quantity, Price, Revenue, Region, Order_Month

## 🎯 Key Features
- ✅ Customer Churn Prediction using Random Forest & Logistic Regression
- ✅ RFM Analysis (Recency, Frequency, Monetary)
- ✅ Feature Engineering — 7 ML features created
- ✅ AI-powered churn explanations using Groq LLM
- ✅ Interactive Power BI Dashboard (4 pages, 15+ visualizations)
- ✅ SQL Analysis with 10 business queries
- ✅ Excel Report with Pivot Tables & Charts
- ✅ Live Streamlit Web App with AI Chatbot
- ✅ Deployed on Streamlit Cloud

## 📁 Project Structure
AI-Powered-Customer-Churn-Prediction/
├── Data/
│   ├── customer_cohort_dataset.csv
│   ├── customer_features.csv
│   ├── AI_Business_Report.txt
│   └── churn_analysis_report.xlsx
├── Notebook/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_model.ipynb
│   └── 05_ai_layer.ipynb
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
├── SQL/
│   └── churn_analysis_queries.sql
├── Power_BI_Dashboard/
│   └── Churn_Analysis_Dashboard.pbix
├── app.py
├── requirements.txt
└── README.md

## 📈 Model Performance
| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| Logistic Regression | ~82% | ~0.85 |
| Random Forest | ~85%+ | ~0.90 |

## 🔍 Key Business Insights
- **24% overall churn rate** across 594 customers
- **Electronics** category contributes highest revenue
- **Recency** is the most important churn predictor
- Customers with **90+ days** since last purchase are high risk
- **East region** has highest churn rate
- **Office** category has lowest churn rate

## 📊 Project Phases
| Phase | Tool | Description |
|-------|------|-------------|
| Phase 1 | Python | Data Loading & EDA |
| Phase 2 | Python | Feature Engineering & Churn Label |
| Phase 3 | Python | ML Model Training & Evaluation |
| Phase 4 | SQL | Business Queries & Analysis |
| Phase 5 | Excel | Pivot Tables & Charts |
| Phase 6 | Power BI | Interactive Dashboard |
| Phase 7 | Python + Groq | AI Layer Integration |
| Phase 8 | Streamlit | Web App Development |
| Phase 9 | GitHub | Version Control & Deployment |

## 🌐 Streamlit App Pages
- **📊 Dashboard** — KPI cards, revenue trends, churn distribution
- **🔍 Churn Predictor** — Real-time churn prediction with AI explanation
- **🤖 AI Assistant** — Ask business questions, generate AI reports
- **📈 Analytics** — Filter by region & category, download data

## 🚀 How to Run Locally
```bash
# Clone repository
git clone https://github.com/Nagendra22-sagar/AI-Powered-Customer-Churn-Prediction.git

# Navigate to folder
cd AI-Powered-Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## 📦 Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
groq

## 🎓 Skills Demonstrated
- **Data Analysis** — EDA, data cleaning, feature engineering
- **Machine Learning** — Classification, model evaluation, hyperparameter tuning
- **SQL** — CTEs, window functions, aggregations, joins
- **Business Intelligence** — Power BI dashboards, KPI reporting
- **Generative AI** — LLM API integration, prompt engineering
- **Web Development** — Streamlit app development and deployment
- **Version Control** — Git & GitHub

## 👨‍💻 Author
**Nagendra V Sagar**
- 📧 nagendravsagar22@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/nagendravsagar)
- 🐙 [GitHub](https://github.com/Nagendra22-sagar)

## 📄 License
This project is open source and available for educational and portfolio purposes.

---
⭐ **If you found this project helpful, please give it a star!** ⭐
