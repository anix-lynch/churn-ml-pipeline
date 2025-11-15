#!/usr/bin/env python3
"""
Simple Churn ML Dashboard - Streamlit Cloud Compatible
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(page_title="Churn ML Dashboard", page_icon="📊")

# Title
st.title("📊 Churn ML Pipeline Dashboard")
st.markdown("### Customer Churn Prediction Demo")

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)

    # Customer data
    customers = pd.DataFrame({
        'customer_id': [f'CUST_{i:04d}' for i in range(100)],
        'age': np.random.normal(35, 10, 100).astype(int).clip(18, 70),
        'country': np.random.choice(['US', 'UK', 'CA', 'DE'], 100),
        'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 100),
        'signup_months': np.random.exponential(24, 100).astype(int)
    })

    return customers

# Mock prediction function
def predict_churn(customer_data):
    """Simple mock prediction based on customer features"""
    risk_score = 0.3  # Base risk

    # Age factor
    if customer_data['age'] > 50:
        risk_score += 0.2
    elif customer_data['age'] < 25:
        risk_score += 0.1

    # Segment factor
    if customer_data['segment'] == 'Basic':
        risk_score += 0.15

    # Tenure factor (longer customers less likely to churn)
    tenure_years = customer_data['signup_months'] / 12
    risk_score -= min(tenure_years * 0.1, 0.2)

    # Add some randomness
    risk_score += np.random.normal(0, 0.1)
    risk_score = np.clip(risk_score, 0.1, 0.9)

    return risk_score

# Sidebar
st.sidebar.header("🎯 Navigation")
page = st.sidebar.radio("Choose a page:", ["Overview", "Predict Churn", "Data Explorer"])

if page == "Overview":
    st.header("📈 Dashboard Overview")

    # Generate sample data
    customers = generate_sample_data()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", len(customers))

    with col2:
        avg_age = customers['age'].mean()
        st.metric("Average Age", ".1f")

    with col3:
        premium_pct = (customers['segment'] == 'Premium').mean() * 100
        st.metric("Premium Customers", ".1f")

    st.subheader("Customer Segments")
    segment_counts = customers['segment'].value_counts()
    st.bar_chart(segment_counts)

    st.subheader("Age Distribution")
    st.hist_chart(customers['age'])

elif page == "Predict Churn":
    st.header("🔮 Churn Prediction")

    # Input form
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 35)
        country = st.selectbox("Country", ["US", "UK", "CA", "DE"])
        segment = st.selectbox("Segment", ["Premium", "Standard", "Basic"])

    with col2:
        signup_months = st.slider("Months Since Signup", 1, 60, 12)

    customer_data = {
        'age': age,
        'country': country,
        'segment': segment,
        'signup_months': signup_months
    }

    if st.button("🎯 Predict Churn Risk", type="primary"):
        with st.spinner("Analyzing customer data..."):
            churn_probability = predict_churn(customer_data)

            st.success("Prediction Complete!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Churn Probability", ".1%")

            with col2:
                risk_level = "High" if churn_probability > 0.6 else "Medium" if churn_probability > 0.4 else "Low"
                color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                st.metric("Risk Level", f"{color} {risk_level}")

            with col3:
                confidence = abs(churn_probability - 0.5) * 200
                st.metric("Confidence", ".1f")

            # Visual gauge
            st.subheader("Risk Assessment")
            if churn_probability > 0.6:
                st.error("🚨 **High Risk**: Immediate attention recommended")
            elif churn_probability > 0.4:
                st.warning("⚠️ **Medium Risk**: Monitor closely")
            else:
                st.success("✅ **Low Risk**: Customer appears stable")

elif page == "Data Explorer":
    st.header("🔍 Data Explorer")

    customers = generate_sample_data()

    st.subheader("Sample Customer Data")
    st.dataframe(customers.head(20))

    st.subheader("Summary Statistics")
    st.write(customers.describe())

    # Country distribution
    st.subheader("Customers by Country")
    country_counts = customers['country'].value_counts()
    st.bar_chart(country_counts)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Churn ML Pipeline Demo")
