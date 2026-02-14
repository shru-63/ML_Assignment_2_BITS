# Machine Learning Assignment 2: Customer Churn Prediction

## a. Problem Statement
The objective of this project is to build and deploy a classification system that predicts whether a telecommunications customer will churn (leave the company) based on their usage patterns and demographic data. This is a binary classification task where we evaluate 6 different models to identify the most accurate predictor for business decision-making.

## b. Dataset Description
- **Source:** [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Instances:** 7,043 (Exceeds the mandatory 500 instances)
- **Features:** 20 initial features, expanded to 30+ after one-hot encoding (Exceeds the mandatory 12 features).
- **Description:** The dataset includes features such as customer tenure, monthly charges, total charges, contract type, and technical support status. The target variable is 'Churn' (Yes/No).

## c. Models Used and Comparison Table
The following 6 models were implemented, trained, and evaluated on the BITS Virtual Lab environment:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8197 | 0.8620 | 0.6831 | 0.5952 | 0.6361 | 0.5192 |
| **Decision Tree** | 0.7119 | 0.6322 | 0.4562 | 0.4611 | 0.4587 | 0.2624 |
| **kNN** | 0.7708 | 0.7895 | 0.5740 | 0.5201 | 0.5457 | 0.3938 |
| **Naive Bayes** | 0.6657 | 0.8377 | 0.4359 | 0.8928 | 0.5858 | 0.4222 |
| **Random Forest** | 0.7913 | 0.8372 | 0.6491 | 0.4611 | 0.5392 | 0.4193 |
| **XGBoost (Ensemble)** | 0.7892 | 0.8392 | 0.6284 | 0.4987 | 0.5561 | 0.4251 |

## d. Performance Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Best Performer:** Achieved the highest Accuracy (81.9%) and MCC (0.519), showing strong balance between classes. |
| **Decision Tree** | **Lowest Performance:** Significant drop in AUC and MCC, likely due to overfitting on the training data. |
| **kNN** | **Moderate:** Performed reasonably well but was sensitive to the high dimensionality of the encoded features. |
| **Naive Bayes** | **Highest Recall:** Best at catching potential churners (89.2% recall) but had a high "false alarm" rate (lowest Precision). |
| **Random Forest** | **Robust:** Showed good stability and high Precision, though slightly outperformed by Logistic Regression in this specific case. |
| **XGBoost** | **Strong Ensemble:** Very competitive AUC (0.839), making it a reliable choice for complex non-linear relationships in the data. |

## e. How to Run
1. Ensure `model/` folder contains all `.pkl` files and the `.ipynb` notebook.
2. Install requirements: `pip install -r requirements.txt`
3. Launch app: `streamlit run app.py`