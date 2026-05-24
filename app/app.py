import streamlit as st
import pandas as pd
import joblib
import os
st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)
st.markdown("""
<style>

/* Main app background */
.stApp{
background: linear-gradient(
135deg,
#0f172a 0%,
#111827 40%,
#1e293b 100%
);

color:white;
}


/* Metric cards */
div[data-testid="metric-container"]{
background: rgba(255,255,255,0.05);

backdrop-filter: blur(10px);

border-radius:15px;

padding:15px;

border:1px solid rgba(255,255,255,0.1);

box-shadow:0px 4px 15px rgba(0,0,0,0.3);
}


/* Sidebar */

section[data-testid="stSidebar"]{
background-color:#111827;
}


/* Button */

.stButton > button{

width:100%;

height:50px;

border-radius:12px;

font-size:18px;

font-weight:bold;

background:linear-gradient(
90deg,
#2563eb,
#7c3aed
);

color:white;

border:none;
}


/* Headings */

h1,h2,h3{
color:white;
}

</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "churn_model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

model_features = joblib.load(
    os.path.join(
        MODEL_DIR,
        "model_features.pkl"
    )
)

st.markdown("""
# 📊Customer Churn Prediction Dashboard

Predict whether a customer is likely to churn using machine learning.

---
""")

# Customer details
st.sidebar.header("Customer Information")

col1,col2 = st.columns(2)

with col1:

    tenure = st.sidebar.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    monthly_charges = st.sidebar.number_input(
        "Monthly Charges",
        value=50.0
    )

    total_charges = st.sidebar.number_input(
        "Total Charges",
        value=500.0
    )

    gender = st.sidebar.selectbox(
        "Gender",
        ["Female","Male"]
    )

    senior = st.sidebar.selectbox(
        "Senior Citizen",
        [0,1]
    )

with col2:

    contract = st.sidebar.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet = st.sidebar.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    tech = st.sidebar.selectbox(
        "Tech Support",
        [
            "Yes",
            "No"
        ]
    )

    security = st.sidebar.selectbox(
        "Online Security",
        [
            "Yes",
            "No"
        ]
    )

    payment = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# Base data
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SeniorCitizen": senior
}

input_df = pd.DataFrame([input_data])

# One-hot encode manually
categorical_inputs = {
    f"gender_{gender}":1,
    f"Contract_{contract}":1,
    f"InternetService_{internet}":1,
    f"TechSupport_{tech}":1,
    f"OnlineSecurity_{security}":1,
    f"PaymentMethod_{payment}":1
}

for key,value in categorical_inputs.items():

    if key in model_features:
        input_df[key] = value

# Fill remaining missing columns
for col in model_features:

    if col not in input_df.columns:
        input_df[col]=0

input_df=input_df[model_features]

input_scaled=scaler.transform(input_df)

if st.button("Predict Churn"):

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]
    st.markdown("---")
    st.header("Prediction Analysis")
    st.progress(float(probability))

    if prediction == 1:

        st.error("High Risk Customer")

    else:

        st.success("Low Risk Customer")


    # Risk category
    if probability < 0.3:

        risk = "Low"

    elif probability < 0.7:

        risk = "Medium"

    else:

        risk = "High"


    st.subheader("Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Risk Level",
            value=risk
        )

    with col2:

        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )


    st.subheader("Recommendations")

    if risk == "High":

        st.write("""
        • Offer promotional discounts

        • Improve customer support

        • Recommend long-term plans

        • Follow up with customer engagement
        """)

    elif risk == "Medium":

        st.write("""
        • Send personalized offers

        • Encourage feature usage

        • Improve retention strategy
        """)

    else:

        st.write("""
        • Maintain customer satisfaction

        • Continue regular engagement
        """)
        st.markdown("---")

st.caption(
    "Built with Streamlit | Random Forest | Scikit-learn"
)