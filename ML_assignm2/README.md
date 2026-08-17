
# Heart Disease Classification

**BITS ID:** 2025AC05939  
**Name:** Kavuri Srilekha  
**Email:** 2025ac05939@wilp.bits-pilani.ac.in  

---

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting the presence of heart disease.

The following five machine learning models were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest

The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

---

## 2. Dataset Description

The Heart Disease dataset contains 918 patient records.

The target variable is:

- `HeartDisease`

Target values:

- `0` = No Heart Disease
- `1` = Heart Disease

The input variables include:

- Age
- Sex
- ChestPainType
- RestingBP
- Cholesterol
- FastingBS
- RestingECG
- MaxHR
- ExerciseAngina
- Oldpeak
- ST_Slope

Categorical variables were converted into numerical representations using one-hot encoding.

Feature scaling was applied for Logistic Regression and kNN.

---

## 3. Machine Learning Models

### Logistic Regression

Logistic Regression was used as a linear classification model.

### Decision Tree

Decision Tree was used to model non-linear relationships between the input features and target variable.

### k-Nearest Neighbors

kNN was implemented with `k = 5`. Feature scaling was applied because kNN is distance-based.

### Naive Bayes

Gaussian Naive Bayes was used as a probabilistic classification model.

### Random Forest

Random Forest was implemented using 100 decision trees.

---

## 4. Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Confusion matrices and classification reports were also generated.

---

## 5. Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8859 | 0.9297 | 0.8716 | 0.9314 | 0.9005 | 0.7694 |
| Decision Tree | 0.7826 | 0.7752 | 0.7818 | 0.8431 | 0.8113 | 0.5580 |
| kNN | 0.8859 | 0.9360 | 0.8857 | 0.9118 | 0.8986 | 0.7686 |
| Naive Bayes | 0.9130 | 0.9451 | 0.9300 | 0.9118 | 0.9208 | 0.8246 |
| Random Forest | 0.8750 | 0.9314 | 0.8835 | 0.8922 | 0.8878 | 0.7468 |

---

## 6. Model-wise Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 0.8859 and an AUC of 0.9297. It obtained the highest recall of 0.9314 among the five models, indicating strong identification of positive heart disease cases.

### Decision Tree

Decision Tree produced the lowest overall performance among the five models, with an accuracy of 0.7826 and an AUC of 0.7752. Its MCC of 0.5580 was also the lowest.

### kNN

kNN achieved an accuracy of 0.8859 and an AUC of 0.9360. Its performance was comparable to Logistic Regression, with slightly higher precision but slightly lower recall.

### Naive Bayes

Naive Bayes achieved the best overall performance. It obtained the highest accuracy (0.9130), AUC (0.9451), precision (0.9300), F1 score (0.9208), and MCC (0.8246).

### Random Forest

Random Forest achieved an accuracy of 0.8750 and an AUC of 0.9314. Its performance was stronger than the Decision Tree but lower than Naive Bayes, Logistic Regression, and kNN on most metrics.

---

## 7. Overall Observation

Naive Bayes achieved the strongest overall performance across the evaluation metrics. It obtained the highest Accuracy, AUC, Precision, F1 Score, and MCC.

Logistic Regression achieved the highest Recall at 0.9314.

Based on the overall comparison, **Naive Bayes was selected as the best-performing model** for this dataset.

---

## 8. Streamlit Application

The Streamlit application provides:

- CSV test-data upload
- Machine learning model selection
- Accuracy display
- AUC display
- Precision display
- Recall display
- F1 Score display
- MCC display
- Confusion matrix
- Classification report

---

## 9. Project Structure

```text
project/
│
├── app.py
├── test_data.csv
├── requirements.txt
├── README.md
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    └── feature_info.json
