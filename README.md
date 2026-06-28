# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Overview

This project uses Machine Learning to predict customer churn in a European retail bank. It helps identify customers who are likely to leave and assigns a churn risk score to support proactive retention strategies.

## Objectives

- Predict customer churn
- Generate churn probability scores
- Identify important churn drivers
- Build an interactive Streamlit dashboard
- Provide explainable AI insights

## Dataset

Dataset: European_Bank.csv

Features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Member
- Estimated Salary
- Exited (Target Variable)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- SHAP
- Streamlit
- Joblib

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost (Optional)

## Project Workflow

1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Feature Importance
8. Churn Risk Scoring
9. Streamlit Deployment

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Streamlit Dashboard Features

- Customer Churn Prediction
- Churn Probability
- Risk Level
- Feature Importance
- What-if Analysis

## Project Structure

```
Bank_Churn_Project/
│
├── European_Bank.csv
├── churn_analysis.ipynb
├── streamlit_app/
│   ├── app.py
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│   └── feature_importance.csv
├── Research Paper.docx
├── Executive Summary.docx
└── README.md
```

## Installation

```bash
git clone https://github.com/sakshamkeshri/Predictive-Modeling-and-Risk-Scoring-for-Bank-Customer-Churn.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
cd streamlit_app
streamlit run app.py
```

## Future Improvements

- Deep Learning Models
- Hyperparameter Tuning
- Live Database Integration
- Customer Retention Recommendation Engine
- Cloud Deployment

## Author

**Saksham Keshri**

MCA Student

Data Analytics & Machine Learning Project
