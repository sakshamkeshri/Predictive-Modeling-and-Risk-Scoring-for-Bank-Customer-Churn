import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
.main{
    background:#f5f7fa;
}

h1{
    color:#003366;
}

.stButton>button{
    width:100%;
    background:#003366;
    color:white;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------

BASE_DIR = Path(__file__).parent

try:
    model = joblib.load(BASE_DIR / "churn_model.pkl")
    columns = joblib.load(BASE_DIR / "columns.pkl")
except Exception as e:
    st.error(f"Model loading error:\n\n{e}")
    st.stop()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🏦 Bank Customer Churn Prediction Dashboard")

st.write(
    "Predict customer churn probability using Machine Learning."
)

# ---------------------------------------------------
# DASHBOARD METRICS
# ---------------------------------------------------

c1, c2, c3 = st.columns(3)

c1.metric("Customers", "10,000")
c2.metric("Churn Rate", "20.37%")
c3.metric("Model", "Random Forest")

st.divider()

st.sidebar.header("Customer Details")

credit_score = st.sidebar.number_input(
    "Credit Score",
    300,
    900,
    650
)

age = st.sidebar.number_input(
    "Age",
    18,
    100,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    value=50000.0
)

products = st.sidebar.selectbox(
    "Number of Products",
    [1,2,3,4]
)

has_cr_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0,1]
)

active = st.sidebar.selectbox(
    "Active Member",
    [0,1]
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    value=50000.0
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

balance_salary_ratio = balance/(salary+1)

product_density = products/(tenure+1)

engagement_product = active*products

age_tenure = age*tenure

geography_germany = 1 if geography=="Germany" else 0
geography_spain = 1 if geography=="Spain" else 0
gender_male = 1 if gender=="Male" else 0


input_df = pd.DataFrame({

    "CreditScore":[credit_score],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[active],
    "EstimatedSalary":[salary],

    "BalanceSalaryRatio":[balance_salary_ratio],
    "ProductDensity":[product_density],
    "EngagementProductInteraction":[engagement_product],
    "AgeTenureInteraction":[age_tenure],

    "Geography_Germany":[geography_germany],
    "Geography_Spain":[geography_spain],
    "Gender_Male":[gender_male]

})

input_df = input_df.reindex(
    columns=columns,
    fill_value=0
)

if st.button("🚀 Predict Churn Risk"):

    probability = model.predict_proba(input_df)[0][1]

    percentage = probability*100

    if probability < 0.30:
        risk="LOW"
        color="green"

    elif probability < 0.60:
        risk="MEDIUM"
        color="orange"

    else:
        risk="HIGH"
        color="red"

    st.success(
        f"Predicted Churn Probability : {percentage:.2f}%"
    )

    st.info(
        f"Risk Category : {risk}"
    )

    fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=percentage,

        title={
            "text":"Customer Churn Risk"
        },

        gauge={

            "axis":{
                "range":[0,100]
            },

            "bar":{
                "color":color
            },

            "steps":[

                {
                    "range":[0,30],
                    "color":"green"
                },

                {
                    "range":[30,60],
                    "color":"yellow"
                },

                {
                    "range":[60,100],
                    "color":"red"
                }

            ]

        }

    ))

    st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("Top Important Features")

try:

    importance = pd.read_csv(
        BASE_DIR/"feature_importance.csv"
    )

    fig2 = px.bar(

        importance.head(10),

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        title="Feature Importance"

    )

    st.plotly_chart(
        fig2,
         width="stretch"
    )

except Exception as e:

    st.warning(
        f"feature_importance.csv not found.\n\n{e}"
    )

st.divider()

st.subheader("Customer Input")

st.dataframe(input_df)

st.caption("Created by Saksham Keshri ")