import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("This app evaluates 6 different ML models to predict if a customer will leave the company.")

# 1. Dataset upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'WA_Fn-UseC_-Telco-Customer-Churn.csv'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # 2. Model selection
    st.sidebar.header("Model Selection")
    model_option = st.sidebar.selectbox(
        "Choose a Machine Learning Model",
        ("logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost")
    )

    if st.sidebar.button("Run Evaluation"):
        try:
            # 3. Preprocessing Logic
            df_clean = df.copy()
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
           
            if 'customerID' in df_clean.columns:
                df_clean.drop('customerID', axis=1, inplace=True)
           
            if 'Churn' in df_clean.columns:
                y_true = df_clean['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
                df_clean.drop('Churn', axis=1, inplace=True)
            else:
                st.error("Target column 'Churn' not found in CSV.")
                st.stop()

            # Perform one-hot encoding
            X = pd.get_dummies(df_clean)

            # --- CRITICAL FIX: Match the exact 30 features your model expects ---
            expected_features = [
                'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
                'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes',
                'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
                'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
                'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes'
            ]
           
            # This reindex forces the app to use only the 30 features the model knows
            X = X.reindex(columns=expected_features, fill_value=0)

            # 4. Load the selected model
            model_path = f"model/{model_option}.pkl"
            model = joblib.load(model_path)
           
            # Predictions
            y_pred = model.predict(X)
           
            # Fix for y_probs: Ensure it is defined for AUC calculation

            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X)[:, 1]
            else:
                y_probs = y_pred

            # 5. Display metrics
            st.subheader(f"📈 Performance Metrics: {model_option.replace('_', ' ').title()}")
           
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
            col2.metric("AUC Score", f"{roc_auc_score(y_true, y_probs):.4f}")
            col3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
            col5.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
            col6.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")

            # 6. Confusion Matrix Visualization
            st.subheader("📉 Visualizations")
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

        except FileNotFoundError:
            st.error(f"Model file '{model_option}.pkl' not found in the 'model/' folder.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload the Telco Churn CSV file from the sidebar to begin.")

