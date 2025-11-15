#!/usr/bin/env python3
"""
Ultra Simple Churn ML Dashboard - Minimal Dependencies
"""
import streamlit as st
import random
from datetime import datetime

# Page config
st.set_page_config(page_title="Churn ML Demo", page_icon="📊")

# Title
st.title("📊 Churn ML Pipeline Demo")
st.markdown("### Customer Churn Prediction - Ultra Simple Version")

# Sidebar
st.sidebar.header("🎯 Navigation")
page = st.sidebar.radio("Choose a page:", ["Overview", "Predict Churn", "About"])

if page == "Overview":
    st.header("📈 Dashboard Overview")

    # Simple metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", "1,000")

    with col2:
        st.metric("Average Age", "35")

    with col3:
        st.metric("Premium Customers", "20%")

    st.success("✅ App is running successfully!")
    st.info("This is a minimal demo to test Streamlit Cloud deployment.")

elif page == "Predict Churn":
    st.header("🔮 Churn Prediction")

    # Simple input form
    st.subheader("Enter Customer Details")

    age = st.slider("Age", 18, 70, 35)
    segment = st.selectbox("Segment", ["Premium", "Standard", "Basic"])
    tenure_months = st.slider("Months Since Signup", 1, 60, 12)

    # Simple prediction logic
    risk_score = 0.3  # Base risk

    if age > 50:
        risk_score += 0.2
    elif age < 25:
        risk_score += 0.1

    if segment == 'Basic':
        risk_score += 0.15

    tenure_years = tenure_months / 12
    risk_score -= min(tenure_years * 0.1, 0.2)

    risk_score += random.uniform(-0.1, 0.1)  # Add some randomness
    risk_score = max(0.05, min(0.95, risk_score))  # Clamp to 0-1

    churn_probability = risk_score
    prediction = "High Risk" if churn_probability > 0.6 else "Medium Risk" if churn_probability > 0.4 else "Low Risk"

    if st.button("🎯 Predict Churn Risk"):
        st.success("Prediction Complete!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Churn Probability", f"{churn_probability:.1%}")

        with col2:
            st.metric("Risk Level", prediction)

        with col3:
            confidence = abs(churn_probability - 0.5) * 200
            st.metric("Confidence", f"{confidence:.1f}")

        # Visual feedback
        if churn_probability > 0.6:
            st.error("🚨 **High Risk**: Immediate attention recommended")
        elif churn_probability > 0.4:
            st.warning("⚠️ **Medium Risk**: Monitor closely")
        else:
            st.success("✅ **Low Risk**: Customer appears stable")

elif page == "About":
    st.header("ℹ️ About This Demo")

    st.markdown("""
    ### Churn ML Pipeline Demo

    This is a simplified demonstration of a customer churn prediction system.

    **Features:**
    - Interactive customer profiling
    - Real-time churn risk assessment
    - Simple ML-based predictions

    **Technology:**
    - Streamlit for the web interface
    - Python for the prediction logic
    - No external dependencies

    **Deployment:**
    - Designed for Streamlit Cloud
    - Minimal resource requirements
    - Fast loading times
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for ML deployment testing")
st.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
