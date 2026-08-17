
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease ML Classifier",
    page_icon="❤️",
    layout="wide"
)

st.title("Heart Disease Classification")
st.write(
    "Compare five machine learning models for Heart Disease classification."
)

# --------------------------------------------------
# Load preprocessing information
# --------------------------------------------------

with open("model/feature_info.json", "r") as f:
    feature_info = json.load(f)

target_column = feature_info["target_column"]
encoded_features = feature_info["encoded_features"]

# --------------------------------------------------
# Load models
# --------------------------------------------------

models = {
    "Logistic Regression": joblib.load(
        "model/logistic_regression.pkl"
    ),
    "Decision Tree": joblib.load(
        "model/decision_tree.pkl"
    ),
    "kNN": joblib.load(
        "model/knn.pkl"
    ),
    "Naive Bayes": joblib.load(
        "model/naive_bayes.pkl"
    ),
    "Random Forest": joblib.load(
        "model/random_forest.pkl"
    )
}

scaler = joblib.load("model/scaler.pkl")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Select a model:",
    list(models.keys())
)

# --------------------------------------------------
# Upload test data
# --------------------------------------------------

st.header("Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(test_df.head())

    if target_column not in test_df.columns:

        st.error(
            f"The uploaded file must contain the target column "
            f"'{target_column}'."
        )

    else:

        # Separate features and target
        X_uploaded = test_df.drop(
            columns=[target_column]
        )

        y_uploaded = test_df[target_column]

        # Encode categorical columns
        X_encoded = pd.get_dummies(
            X_uploaded,
            drop_first=True
        )

        # Match training feature columns
        X_encoded = X_encoded.reindex(
            columns=encoded_features,
            fill_value=0
        )

        # Select model
        model = models[selected_model]

        # Scale only models that require scaling
        if selected_model in [
            "Logistic Regression",
            "kNN"
        ]:
            X_model = scaler.transform(X_encoded)
        else:
            X_model = X_encoded

        # Predictions
        predictions = model.predict(X_model)
        probabilities = model.predict_proba(X_model)[:, 1]

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        accuracy = accuracy_score(
            y_uploaded,
            predictions
        )

        auc = roc_auc_score(
            y_uploaded,
            probabilities
        )

        precision = precision_score(
            y_uploaded,
            predictions
        )

        recall = recall_score(
            y_uploaded,
            predictions
        )

        f1 = f1_score(
            y_uploaded,
            predictions
        )

        mcc = matthews_corrcoef(
            y_uploaded,
            predictions
        )

        st.header("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        st.header("Confusion Matrix")

        cm = confusion_matrix(
            y_uploaded,
            predictions
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        st.dataframe(cm_df)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------

        st.header("Classification Report")

        report = classification_report(
            y_uploaded,
            predictions,
            target_names=[
                "No Disease",
                "Disease"
            ],
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4)
        )

else:

    st.info(
        "Please upload test_data.csv to evaluate the selected model."
    )
