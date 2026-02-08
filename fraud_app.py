import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px


def generate_transactions(n_samples=2000, fraud_ratio=0.05):
    """Generate dummy credit card transactions with injected fraud anomalies."""
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit_amounts = np.random.exponential(50, n_legit) + 10
    legit_distances = np.random.exponential(10, n_legit)
    legit_hours = np.random.normal(14, 4, n_legit).clip(0, 23)

    # Fraud transactions (high amounts, far distances, odd hours)
    fraud_amounts = np.random.exponential(200, n_fraud) + 500
    fraud_distances = np.random.exponential(50, n_fraud) + 100
    fraud_hours = np.random.choice([2, 3, 4, 23, 0, 1], n_fraud)

    df = pd.DataFrame({
        "Transaction_Amount": np.concatenate([legit_amounts, fraud_amounts]),
        "Distance_From_Home": np.concatenate([legit_distances, fraud_distances]),
        "Transaction_Hour": np.concatenate([legit_hours, fraud_hours]),
        "is_fraud": [0] * n_legit + [1] * n_fraud
    })

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@st.cache_resource
def train_model(df):
    """Train IsolationForest model on transaction data."""
    X = df[["Transaction_Amount", "Distance_From_Home", "Transaction_Hour"]]
    model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    model.fit(X)
    return model


def main():
    st.set_page_config(page_title="Flash Fraud Detector", page_icon="⚡", layout="wide")
    st.title("⚡ Flash Fraud Detector")

    # Generate data and train model
    df = generate_transactions()
    model = train_model(df)

    # Sidebar inputs
    st.sidebar.header("🔍 New Transaction")
    amount = st.sidebar.slider(
        "Transaction Amount ($)",
        min_value=1.0,
        max_value=1000.0,
        value=50.0,
        step=1.0
    )
    distance = st.sidebar.slider(
        "Distance From Home (miles)",
        min_value=0.0,
        max_value=200.0,
        value=10.0,
        step=0.5
    )
    hour = st.sidebar.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    # Analyze button
    if st.sidebar.button("🔎 Analyze", type="primary", use_container_width=True):
        user_input = np.array([[amount, distance, hour]])
        prediction = model.predict(user_input)[0]

        if prediction == -1:
            st.sidebar.error("## 🔴 FRAUD")
            st.sidebar.warning("This transaction shows anomalous patterns!")
        else:
            st.sidebar.success("## 🟢 LEGIT")
            st.sidebar.info("This transaction appears normal.")

    # Create scatter plot
    st.subheader("📊 Transaction Overview")

    plot_df = df.copy()
    plot_df["Type"] = plot_df["is_fraud"].map({0: "Legit", 1: "Fraud"})

    # Add user input point
    user_point = pd.DataFrame({
        "Transaction_Amount": [amount],
        "Distance_From_Home": [distance],
        "Transaction_Hour": [hour],
        "Type": ["Your Input"]
    })
    plot_df = pd.concat([plot_df, user_point], ignore_index=True)

    fig = px.scatter(
        plot_df,
        x="Distance_From_Home",
        y="Transaction_Amount",
        color="Type",
        color_discrete_map={"Legit": "#2ecc71", "Fraud": "#e74c3c", "Your Input": "#f39c12"},
        symbol="Type",
        symbol_map={"Legit": "circle", "Fraud": "circle", "Your Input": "star"},
        size_max=15,
        hover_data=["Transaction_Hour"],
        title="Transactions: Amount vs Distance from Home"
    )

    # Make user input point larger
    fig.update_traces(
        marker=dict(size=12),
        selector=dict(name="Your Input")
    )
    fig.update_traces(
        marker=dict(size=6, opacity=0.6),
        selector=dict(name="Legit")
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.8),
        selector=dict(name="Fraud")
    )

    fig.update_layout(
        xaxis_title="Distance From Home (miles)",
        yaxis_title="Transaction Amount ($)",
        legend_title="Transaction Type",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        st.metric("Fraud Rate", f"{df['is_fraud'].mean() * 100:.1f}%")
    with col3:
        st.metric("Model", "IsolationForest")


if __name__ == "__main__":
    main()
