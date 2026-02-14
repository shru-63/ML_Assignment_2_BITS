import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("This app evaluates 6 different ML models to predict if a customer will leave the company.")

# 1. Dataset upload option (CSV)
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'WA_Fn-UseC_-Telco-Customer-Churn.csv'", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # 2. Model selection dropdown
    st.sidebar.header("Model Selection")
    model_option = st.sidebar.selectbox(
        "Choose a Machine Learning Model",
        ("logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost")
    )

    if st.sidebar.button("Run Evaluation"):
        try:
            # 3. Preprocessing Logic (Must match your notebook)
            df_clean = df.copy()
            # Convert TotalCharges to numeric and handle missing values
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
            
            # Drop customerID as it's not a feature
            if 'customerID' in df_clean.columns:
                df_clean.drop('customerID', axis=1, inplace=True)
            
            # Separate target and features
            if 'Churn' in df_clean.columns:
                y_true = df_clean['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
                df_clean.drop('Churn', axis=1, inplace=True)
            else:
                st.error("Target column 'Churn' not found in CSV.")
                st.stop()

            # One-hot encoding (get_dummies)
            X = pd.get_dummies(df_clean)

            # 4. Load the selected model
            model_path = f"model/{model_option}.pkl"
            model = joblib.load(model_path)
           
            # --- FIX STARTS HERE ---
            # Try to get features from the model, otherwise fall back to the columns in X
            if hasattr(model, "feature_names_in_"):
                model_features = model.feature_names_in_
                X = X.reindex(columns=model_features, fill_value=0)
            else:
                # If the model doesn't have feature_names_in_, we assume the
                # preprocessing in this app matches the training.
                st.warning("Model does not contain feature names; using current data columns.")
            # --- FIX ENDS HERE ---

            # Predictions
            y_pred = model.predict(X)

            
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
