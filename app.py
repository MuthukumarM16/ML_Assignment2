from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import evaluate_model, load_dataset


ROOT = Path(__file__).parent
MODEL_BUNDLE_PATH = ROOT / "model" / "model_bundle.joblib"
TEST_DATA_PATH = ROOT / "test_data.csv"


@st.cache_resource
def load_bundle():
    if not MODEL_BUNDLE_PATH.exists():
        import train_models

        train_models.main()
    return joblib.load(MODEL_BUNDLE_PATH)


@st.cache_data
def load_default_test_data():
    return pd.read_csv(TEST_DATA_PATH)


def split_features_target(dataframe, feature_names):
    if "target" not in dataframe.columns:
        raise ValueError("CSV must contain a 'target' column for evaluation.")

    missing_features = [column for column in feature_names if column not in dataframe.columns]
    if missing_features:
        raise ValueError(
            "CSV is missing required feature columns: " + ", ".join(missing_features[:5])
        )

    return dataframe[feature_names], dataframe["target"]


def format_metric_table(dataframe):
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    return dataframe.style.format({column: "{:.4f}" for column in metric_columns})


def confusion_matrix_block(matrix, labels):
    max_value = max(int(matrix.max()), 1)
    rows = []
    for row_index, actual_label in enumerate(labels):
        cells = []
        for column_index, predicted_label in enumerate(labels):
            value = int(matrix[row_index][column_index])
            intensity = 0.12 + 0.78 * (value / max_value)
            text_color = "#FFFFFF" if intensity > 0.45 else "#0F172A"
            cell_label = (
                f"Actual {actual_label}<br>Predicted {predicted_label}"
            )
            cells.append(
                f"""
                <div class="cm-cell" style="background: rgba(37, 99, 235, {intensity:.2f}); color: {text_color};">
                    <div class="cm-value">{value}</div>
                    <div class="cm-label">{cell_label}</div>
                </div>
                """
            )
        rows.append("".join(cells))

    return f"""
    <style>
    body {{
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #0F172A;
    }}
    .cm-wrap {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }}
    .cm-title {{
        color: #0F172A;
        font-size: 1rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 8px;
    }}
    .cm-axis {{
        color: #475569;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .cm-axis-top {{
        text-align: center;
        margin-bottom: 8px;
    }}
    .cm-layout {{
        display: grid;
        grid-template-columns: 34px 1fr;
        gap: 10px;
        align-items: center;
    }}
    .cm-axis-left {{
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        text-align: center;
    }}
    .cm-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(130px, 1fr));
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        overflow: hidden;
    }}
    .cm-cell {{
        min-height: 128px;
        border: 1px solid rgba(255, 255, 255, 0.65);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 14px;
        box-sizing: border-box;
    }}
    .cm-value {{
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }}
    .cm-label {{
        font-size: 0.82rem;
        font-weight: 650;
        margin-top: 10px;
        opacity: 0.9;
    }}
    </style>
    <div class="cm-wrap">
        <div class="cm-title">Confusion Matrix Block Diagram</div>
        <div class="cm-axis cm-axis-top">Predicted</div>
        <div class="cm-layout">
            <div class="cm-axis cm-axis-left">Actual</div>
            <div class="cm-grid">
                {''.join(rows)}
            </div>
        </div>
    </div>
    """


st.set_page_config(
    page_title="Spambase Email Classification Lab",
    page_icon=":email:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricLabel"] p {
        color: #475569;
        font-size: 0.86rem;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-size: 1.55rem;
    }
    section[data-testid="stSidebar"] {
        background: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    .hero {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1.1rem;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        font-size: 2.1rem;
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #475569;
        font-size: 1rem;
        margin: 0;
    }
    .section-title {
        color: #0F172A;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 1.1rem 0 0.65rem;
    }
    .status-pill {
        display: inline-block;
        border: 1px solid #BBF7D0;
        background: #F0FDF4;
        color: #166534;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        font-size: 0.82rem;
        font-weight: 650;
        margin-top: 0.75rem;
    }
    .result-section {
        border-top: 1px solid #E2E8F0;
        padding-top: 1.1rem;
        margin-top: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

bundle = load_bundle()
feature_names = bundle["feature_names"]
target_names = bundle["target_names"]
target_labels = [str(label) for label in target_names]
models = bundle["models"]

dataset, _, _ = load_dataset()
default_data = load_default_test_data()
comparison_df = bundle["metrics"].copy()

with st.sidebar:
    st.header("Controls")
    selected_model_name = st.selectbox("Model", list(models.keys()))
    uploaded_file = st.file_uploader("Test CSV", type=["csv"])
    st.divider()
    st.caption("Expected CSV")
    st.write(f"{len(feature_names)} feature columns plus `target`")
    st.caption("Target labels")
    st.write("`0 = not_spam`, `1 = spam`")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    data_source = "Uploaded test data"
else:
    test_data = default_data
    data_source = "Included test split"

st.markdown(
    """
    <div class="hero">
        <h1>Spambase Email Classification Lab</h1>
        <p>Compare five classic machine learning classifiers on spam email detection using the UCI Spambase dataset.</p>
        <span class="status-pill">Deployment-ready Streamlit app</span>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_cols = st.columns(4)
summary_cols[0].metric("Dataset Rows", f"{dataset.shape[0]:,}")
summary_cols[1].metric("Features", f"{len(feature_names)}")
summary_cols[2].metric("Test Rows", f"{test_data.shape[0]:,}")
summary_cols[3].metric("Classes", ", ".join(target_labels))

try:
    x_test, y_test = split_features_target(test_data, feature_names)
    model = models[selected_model_name]
    predictions = model.predict(x_test)
    metrics = evaluate_model(model, x_test, y_test)

    st.markdown('<div class="section-title">Selected Model Performance</div>', unsafe_allow_html=True)
    st.caption(f"{selected_model_name} evaluated on {data_source.lower()}")

    metric_cols = st.columns(6)
    for metric_col, (label, value) in zip(metric_cols, metrics.items()):
        metric_col.metric(label, f"{value:.4f}")

    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.bar_chart(
        comparison_df,
        x="ML Model Name",
        y=["Accuracy", "AUC", "F1", "MCC"],
        height=280,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=target_labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    matrix = confusion_matrix(y_test, predictions)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual {name}" for name in target_labels],
        columns=[f"Predicted {name}" for name in target_labels],
    )

    st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    st.dataframe(report_df.style.format("{:.4f}"), width="stretch")

    st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    matrix_visual_col, matrix_table_col = st.columns([1, 1])
    with matrix_visual_col:
        st.html(confusion_matrix_block(matrix, target_labels))
    with matrix_table_col:
        st.dataframe(matrix_df, width="stretch")

    st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Test Data</div>', unsafe_allow_html=True)
    preview_cols = ["target"] + feature_names[:8]
    st.dataframe(test_data[preview_cols], width="stretch", height=420)

    st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">All Metrics</div>', unsafe_allow_html=True)
    st.dataframe(format_metric_table(comparison_df), width="stretch")
except Exception as exc:
    st.error(str(exc))
