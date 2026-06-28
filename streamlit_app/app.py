import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    color: #003366;
}

.metric-card {
    background-color:white;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("churn_model.pkl")

st.title("🏦 Bank Customer Churn Prediction Dashboard")

st.markdown(
    "AI Powered Customer Retention Intelligence System"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Customers",
        value="10,000"
    )

with col2:
    st.metric(
        label="Churn Rate",
        value="20.37%"
    )

with col3:
    st.metric(
        label="Model",
        value="XGBoost"
    )

st.divider()

st.sidebar.header("Customer Information")

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
    "Is Active Member",
    [0,1]
)

estimated_salary = st.sidebar.number_input(
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

geography_germany = 0
geography_spain = 0

if geography == "Germany":
    geography_germany = 1

elif geography == "Spain":
    geography_spain = 1

gender_male = 1 if gender == "Male" else 0


balance_salary_ratio = (
    balance/(estimated_salary+1)
)

product_density = (
    products/(age+1)
)

engagement_product = (
    active*products
)

age_tenure = (
    age*tenure
)

input_df = pd.DataFrame({

    "CreditScore":[credit_score],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[active],
    "EstimatedSalary":[estimated_salary],

    "BalanceSalaryRatio":[balance_salary_ratio],
    "ProductDensity":[product_density],
    "EngagementProduct":[engagement_product],
    "AgeTenure":[age_tenure],

    "Geography_Germany":[geography_germany],
    "Geography_Spain":[geography_spain],

    "Gender_Male":[gender_male]
})


if st.button("🚀 Predict Churn Risk"):

    probability = model.predict_proba(
        input_df
    )[0][1]

    percentage = probability * 100

    if probability < 0.30:
        risk = "LOW"
        color = "green"

    elif probability < 0.60:
        risk = "MEDIUM"
        color = "orange"

    else:
        risk = "HIGH"
        color = "red"

    st.success(
        f"Predicted Churn Probability: {percentage:.2f}%"
    )

    st.warning(
        f"Risk Category: {risk}"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        title={'text': "Churn Risk Score"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': color},
            'steps': [
                {'range':[0,30],'color':'green'},
                {'range':[30,60],'color':'yellow'},
                {'range':[60,100],'color':'red'}
            ]
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.subheader("📊 Feature Importance")

try:

    importance = pd.read_csv(
        "feature_importance.csv"
    )

    fig2 = px.bar(
        importance.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        title="Top Features Influencing Churn"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

except:
    st.info(
        "feature_importance.csv not found."
    )

st.subheader("🔍 What-If Scenario Analysis")

st.markdown("""
Modify customer attributes from the sidebar and click **Predict Churn Risk**
to observe how churn probability changes.
""")

st.divider()

