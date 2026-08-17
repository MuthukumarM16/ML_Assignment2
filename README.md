# Spam Email Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare multiple supervised machine learning classification models for identifying whether an email is spam or not spam. Each record represents an email summarized through word-frequency, character-frequency, and capital-letter sequence features. The project includes model training, evaluation using standard classification metrics, and an interactive Streamlit web application where test data can be uploaded and model performance can be viewed.

## b. Dataset Description

This project uses the Spambase dataset, a public classification dataset from the UCI Machine Learning Repository.

- Number of instances: 4,601
- Number of input features: 57
- Target classes: Not spam and spam
- Problem type: Binary classification
- Feature type: Numeric word-frequency, character-frequency, and capital-run-length attributes
- Target meaning: `0` means not spam and `1` means spam

The dataset satisfies the assignment constraints because it has more than 500 instances and more than 12 features.

## c. GitHub Repository Link

GitHub Repository Link: [ML_Assignment2](https://github.com/MuthukumarM16/ML_Assignment2)

Live Streamlit App Link: [Spambase Email Classification App](https://mlassignment2-xpdyarvzmwdeevknnpesn6.streamlit.app/)

## d. Models Used

The following models were trained and evaluated on the same stratified test split:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor Classifier
- Gaussian Naive Bayes Classifier
- Random Forest Classifier

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9279 | 0.9728 | 0.9304 | 0.8833 | 0.9062 | 0.8485 |
| Decision Tree | 0.9062 | 0.9323 | 0.8950 | 0.8634 | 0.8789 | 0.8027 |
| kNN | 0.9053 | 0.9568 | 0.8929 | 0.8634 | 0.8779 | 0.8009 |
| Naive Bayes | 0.8306 | 0.9474 | 0.7133 | 0.9537 | 0.8162 | 0.6893 |
| Random Forest (Ensemble) | 0.9322 | 0.9826 | 0.9332 | 0.8921 | 0.9122 | 0.8576 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression performed strongly with high accuracy, AUC, precision, F1 score, and MCC. This indicates that many spam patterns in the dataset can be separated well using a linear decision boundary after feature scaling. |
| Decision Tree | The Decision Tree gave good performance but was weaker than Logistic Regression and Random Forest. A single tree can capture non-linear rules, but it is more sensitive to the train-test split and individual feature thresholds. |
| kNN | kNN achieved performance close to the Decision Tree. Since the dataset has 57 numeric features, distance-based classification works reasonably well after scaling, but it did not outperform the stronger linear and ensemble models. |
| Naive Bayes | Naive Bayes achieved the highest recall, meaning it detected most spam emails. However, its precision was much lower, so it also marked more non-spam emails as spam. This is important because false positives are undesirable in email filtering. |
| Random Forest (Ensemble) | Random Forest achieved the best overall performance with the highest accuracy, AUC, F1 score, and MCC. The ensemble model handled non-linear feature interactions better than a single Decision Tree. |
| Overall Winner for this dataset | Random Forest is the overall winner because it achieved the best balance across accuracy, AUC, precision, recall, F1 score, and MCC. |

## Streamlit Application Features

The Streamlit application includes:

- CSV test data upload option
- Model selection dropdown
- Evaluation metrics display
- Classification report
- Confusion matrix
- Test data preview
- Overall model comparison table

## Repository Structure

```text
project-folder/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- metrics.csv
|-- data/
|   |-- spambase.data
|   |-- spambase.names
|   |-- spambase.csv
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest.joblib
|   |-- model_bundle.joblib
```

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Deployment Steps

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Select the main branch.
5. Set the app file path to `app.py`.
6. Deploy the app.

## Notes

The included `test_data.csv` file contains the test split used for evaluation. It includes all 57 feature columns and the `target` column required by the Streamlit app for metric calculation.

Dataset source: UCI Machine Learning Repository, Spambase dataset.
