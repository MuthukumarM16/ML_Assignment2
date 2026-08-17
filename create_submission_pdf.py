from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PATH = OUTPUT_DIR / "README.pdf"
SCREENSHOT_PATH = Path(
    "/var/folders/42/v0c8c0jx2d17vxs7z08_gz0c0000gn/T/"
    "codex-clipboard-c6774700-a163-4e07-a122-ac31af19c04d.png"
)
GITHUB_URL = "https://github.com/MuthukumarM16/ML_Assignment2"
STREAMLIT_URL = "https://mlassignment2-xpdyarvzmwdeevknnpesn6.streamlit.app/"


def paragraph(text, style):
    return Paragraph(text.replace("&", "&amp;"), style)


def build_pdf():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = styles["Title"]
    heading = styles["Heading2"]
    normal = styles["BodyText"]
    normal.spaceAfter = 8
    small = ParagraphStyle(
        "Small",
        parent=normal,
        fontSize=8,
        leading=10,
    )
    placeholder = ParagraphStyle(
        "Placeholder",
        parent=normal,
        textColor=colors.HexColor("#1D4ED8"),
        backColor=colors.HexColor("#EFF6FF"),
        borderColor=colors.HexColor("#93C5FD"),
        borderPadding=5,
        leading=12,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=LETTER,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    elements = [
        paragraph("Machine Learning Assignment 2", title),
        paragraph("Spam Email Classification Using Machine Learning", heading),
        paragraph(
            "This PDF contains the required GitHub repository link, live Streamlit app link, "
            "BITS Virtual Lab execution screenshot, README content, model comparison table, "
            "model observations, and final submission checklist.",
            normal,
        ),
        paragraph(
            f'GitHub Repository Link: <link href="{GITHUB_URL}">{GITHUB_URL}</link>',
            placeholder,
        ),
        paragraph(
            f'Live Streamlit App Link: <link href="{STREAMLIT_URL}">{STREAMLIT_URL}</link>',
            placeholder,
        ),
        paragraph("BITS Virtual Lab Screenshot", heading),
    ]
    if SCREENSHOT_PATH.exists():
        screenshot = Image(str(SCREENSHOT_PATH))
        screenshot._restrictSize(6.35 * inch, 3.75 * inch)
        elements.extend([screenshot, Spacer(1, 10)])
    else:
        elements.append(
            paragraph("Screenshot file was not available while generating this PDF.", normal)
        )
    elements.extend(
        [
        Spacer(1, 10),
        paragraph("a. Problem Statement", heading),
        paragraph(
            "The objective of this project is to build and compare multiple supervised machine "
            "learning classification models for identifying whether an email is spam or not "
            "spam. Each record represents an email summarized through word-frequency, "
            "character-frequency, and capital-letter sequence features. The project includes "
            "model training, evaluation using standard classification metrics, and an interactive "
            "Streamlit web application where test data can be uploaded and model performance can "
            "be viewed.",
            normal,
        ),
        paragraph("b. Dataset Description", heading),
        paragraph(
            "This project uses the Spambase dataset, a public classification dataset from the "
            "UCI Machine Learning Repository.",
            normal,
        ),
        ]
    )

    dataset_rows = [
        ["Property", "Value"],
        ["Number of instances", "4,601"],
        ["Number of input features", "57"],
        ["Target classes", "Not spam and spam"],
        ["Problem type", "Binary classification"],
        [
            "Feature type",
            "Numeric word-frequency, character-frequency, and capital-run-length attributes",
        ],
        ["Target meaning", "0 means not spam and 1 means spam"],
    ]
    elements.append(make_table(dataset_rows, [1.7 * inch, 4.6 * inch], small))
    elements.extend(
        [
            paragraph(
                "The dataset satisfies the assignment constraints because it has more than 500 "
                "instances and more than 12 features.",
                normal,
            ),
            paragraph("c. GitHub Repository Link", heading),
            paragraph(
                f'GitHub Repository Link: <link href="{GITHUB_URL}">{GITHUB_URL}</link>',
                placeholder,
            ),
            paragraph(
                f'Live Streamlit App Link: <link href="{STREAMLIT_URL}">{STREAMLIT_URL}</link>',
                placeholder,
            ),
            paragraph("d. Models Used", heading),
            paragraph(
                "The following models were trained and evaluated on the same stratified test split: "
                "Logistic Regression, Decision Tree Classifier, K-Nearest Neighbor Classifier, "
                "Gaussian Naive Bayes Classifier, and Random Forest Classifier.",
                normal,
            ),
            paragraph("Model Comparison Table", heading),
        ]
    )

    metric_rows = [
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        ["Logistic Regression", "0.9279", "0.9728", "0.9304", "0.8833", "0.9062", "0.8485"],
        ["Decision Tree", "0.9062", "0.9323", "0.8950", "0.8634", "0.8789", "0.8027"],
        ["kNN", "0.9053", "0.9568", "0.8929", "0.8634", "0.8779", "0.8009"],
        ["Naive Bayes", "0.8306", "0.9474", "0.7133", "0.9537", "0.8162", "0.6893"],
        ["Random Forest (Ensemble)", "0.9322", "0.9826", "0.9332", "0.8921", "0.9122", "0.8576"],
    ]
    elements.append(make_table(metric_rows, [1.45 * inch, 0.8 * inch, 0.65 * inch, 0.8 * inch, 0.7 * inch, 0.6 * inch, 0.6 * inch], small))
    elements.extend([PageBreak(), paragraph("Model Performance Observations", heading)])

    observation_rows = [
        ["ML Model Name", "Observation about model performance"],
        [
            "Logistic Regression",
            "Performed strongly with high accuracy, AUC, precision, F1 score, and MCC. Many spam patterns can be separated well using a linear decision boundary after feature scaling.",
        ],
        [
            "Decision Tree",
            "Gave good performance but was weaker than Logistic Regression and Random Forest. A single tree is sensitive to the train-test split and individual feature thresholds.",
        ],
        [
            "kNN",
            "Achieved performance close to the Decision Tree. Distance-based classification worked reasonably well after scaling but did not outperform the stronger models.",
        ],
        [
            "Naive Bayes",
            "Achieved the highest recall, meaning it detected most spam emails, but lower precision means it also marked more non-spam emails as spam.",
        ],
        [
            "Random Forest (Ensemble)",
            "Achieved the best overall performance with the highest accuracy, AUC, F1 score, and MCC. The ensemble handled non-linear feature interactions well.",
        ],
        [
            "Overall Winner for this dataset",
            "Random Forest is the overall winner because it achieved the best balance across all required metrics.",
        ],
    ]
    elements.append(make_table(observation_rows, [1.7 * inch, 4.6 * inch], small))
    elements.extend(
        [
            Spacer(1, 10),
            paragraph("Streamlit Application Features", heading),
            paragraph(
                "The Streamlit application includes CSV test data upload, model selection dropdown, "
                "evaluation metrics display, classification report, confusion matrix, test data "
                "preview, and an overall model comparison table.",
                normal,
            ),
            paragraph("Repository Structure", heading),
            paragraph(
                "The repository contains app.py, train_models.py, requirements.txt, README.md, "
                "test_data.csv, metrics.csv, data/spambase.data, data/spambase.names, "
                "data/spambase.csv, and saved model artifacts under the model folder.",
                normal,
            ),
            paragraph("Final Submission Checklist", heading),
        ]
    )
    checklist_rows = [
        ["Checklist Item", "Status"],
        ["GitHub repo link works", "Done"],
        ["Streamlit app link opens correctly", "Done"],
        ["App loads without errors", "Done"],
        ["All required Streamlit features implemented", "Done"],
        ["README.md updated", "Done"],
        ["README content added in submitted PDF", "Done"],
        ["BITS Virtual Lab screenshot included", "Done"],
    ]
    elements.append(make_table(checklist_rows, [4.7 * inch, 1.6 * inch], small))

    doc.build(elements)
    print(OUTPUT_PATH)


def make_table(rows, col_widths, style):
    formatted_rows = []
    for row in rows:
        formatted_rows.append([paragraph(str(cell), style) for cell in row])
    table = Table(formatted_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


if __name__ == "__main__":
    build_pdf()
