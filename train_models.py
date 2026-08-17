from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "spambase.data"
FULL_DATA_PATH = DATA_DIR / "spambase.csv"
TEST_DATA_PATH = ROOT / "test_data.csv"
METRICS_PATH = ROOT / "metrics.csv"
MODEL_BUNDLE_PATH = MODEL_DIR / "model_bundle.joblib"
RANDOM_STATE = 42

FEATURE_NAMES = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_semicolon",
    "char_freq_parenthesis",
    "char_freq_bracket",
    "char_freq_exclamation",
    "char_freq_dollar",
    "char_freq_hash",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]
TARGET_NAME = "target"
TARGET_NAMES = ["not_spam", "spam"]


def load_dataset():
    columns = FEATURE_NAMES + [TARGET_NAME]
    frame = pd.read_csv(RAW_DATA_PATH, header=None, names=columns)
    features = frame[FEATURE_NAMES]
    target = frame[TARGET_NAME]
    return frame, features, target


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def positive_class_scores(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def evaluate_model(model, features, target):
    predictions = model.predict(features)
    scores = positive_class_scores(model, features)
    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, scores),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    frame, features, target = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    test_data = x_test.copy()
    test_data["target"] = y_test
    test_data.to_csv(TEST_DATA_PATH, index=False)
    frame.to_csv(FULL_DATA_PATH, index=False)

    trained_models = {}
    metric_rows = []
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        metrics = evaluate_model(model, x_test, y_test)
        metric_rows.append({"ML Model Name": name, **metrics})
        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(METRICS_PATH, index=False)

    joblib.dump(
        {
            "models": trained_models,
            "feature_names": list(features.columns),
            "target_names": TARGET_NAMES,
            "metrics": metrics_df,
        },
        MODEL_BUNDLE_PATH,
    )

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
