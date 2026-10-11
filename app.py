
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.joblib")

st.title("🏠 House Price Prediction System")
st.write(
    "Estimate residential house prices in selected Nigerian locations."
)

try:
    saved = load_model()
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    location = st.selectbox(
        "Location",
        ["Damaturu", "Maiduguri", "Kano", "Abuja"]
    )
    bedrooms = st.number_input(
        "Number of bedrooms", 1, 20, 3
    )
    bathrooms = st.number_input(
        "Number of bathrooms", 1, 15, 2
    )
    size = st.number_input(
        "House size (square metres)", 20, 2000, 150
    )
    age = st.number_input(
        "House age (years)", 0, 150, 5
    )

    if st.button("Predict House Price"):
        row = pd.DataFrame([{
            "Location": location,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Size_sqm": size,
            "House_Age": age
        }])

        row = pd.get_dummies(
            row, columns=["Location"], dtype=int
        )
        row = row.reindex(
            columns=feature_columns, fill_value=0
        )

        prediction = model.predict(row)[0]

        st.success(
            f"Estimated house price: ₦{prediction:,.0f}"
        )
        st.caption(
            "This is an estimate based on simulated training data, "
            "not a verified market valuation."
        )

except FileNotFoundError:
    st.error(
        "The trained model file is missing. "
        "Upload house_price_model.joblib to this repository."
    )
  
