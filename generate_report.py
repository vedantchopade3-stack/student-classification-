import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


# ==================================================
# PATH CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

METADATA_FILE = MODEL_DIR / "model_metadata.json"
PDF_FILE = REPORT_DIR / "Student_Model_Evaluation_Report.pdf"

REPORT_DIR.mkdir(exist_ok=True)


# ==================================================
# LOAD MODEL METADATA
# ==================================================

def load_metadata():

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Model metadata not found. Run python train_model.py first."
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# ==================================================
# FORMAT METRICS
# ==================================================

def safe_float(value):

    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def get_classification_report(metadata):

    report = metadata.get("classification_report", {})

    rows = [
        [
            "Class",
            "Precision",
            "Recall",
            "F1 Score",
            "Support"
        ]
    ]

    if not isinstance(report, dict):
        return rows

    for label, values in report.items():

        if not isinstance(values, dict):
            continue

        if label in ("accuracy", "macro avg", "weighted avg"):
            continue

        rows.append([
            str(label),
            f"{safe_float(values.get('precision')):.4f}",
            f"{safe_float(values.get('recall')):.4f}",
            f"{safe_float(values.get('f1-score')):.4f}",
            str(values.get("support", 0))
        ])

    for label in ("macro avg", "weighted avg"):

        values = report.get(label)

        if isinstance(values, dict):

            rows.append([
                label,
                f"{safe_float(values.get('precision')):.4f}",
                f"{safe_float(values.get('recall')):.4f}",
                f"{safe_float(values.get('f1-score')):.4f}",
                str(values.get("support", 0))
            ])

    return rows


# ==================================================
# CONFUSION MATRIX
# ==================================================

def get_confusion_matrix(metadata):

    matrix = metadata.get("confusion_matrix", [])
    classes = metadata.get("classes", [])

    if not matrix:
        return None

    if not isinstance(matrix, list):
        return None

    header = ["Actual / Predicted"]

    if classes:
        header.extend([str(x) for x in classes])
    else:
        header.extend([
            f"Class {i + 1}"
            for i in range(len(matrix[0]))
        ])

    rows = [header]

    for index, row in enumerate(matrix):

        if classes and index < len(classes):
            label = str(classes[index])
        else:
            label = f"Class {index + 1}"

        rows.append([label] + [str(x) for x in row])

    return rows


# ==================================================
# PDF STYLES
# ==================================================

def get_styles():

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=25,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=12
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=14,
            spaceAfter=8
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["Normal"],
            fontSize=9,
            leading=13
        )
    )

    return styles


# ==================================================
# TABLE FORMAT
# ==================================================

def format_table(data, widths=None):

    table = Table(
        data,
        colWidths=widths,
        repeatRows=1
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#17365D")),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),

            ("FONTSIZE", (0, 0), (-1, -1), 8),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("ALIGN", (1, 1), (-1, -1), "CENTER"),

            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#EAF0F8")]),

            ("TOPPADDING", (0, 0), (-1, -1), 7),

            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),

            ("LEFTPADDING", (0, 0), (-1, -1), 6),

            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    return table


# ==================================================
# PAGE HEADER AND FOOTER
# ==================================================

def add_page_number(canvas, document):

    canvas.saveState()

    page_width, page_height = A4

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)

    canvas.drawString(
        0.65 * inch,
        0.45 * inch,
        "Student Classification System | Model Evaluation"
    )

    canvas.drawRightString(
        page_width - 0.65 * inch,
        0.45 * inch,
        f"Page {document.page}"
    )

    canvas.setStrokeColor(colors.lightgrey)

    canvas.line(
        0.65 * inch,
        0.60 * inch,
        page_width - 0.65 * inch,
        0.60 * inch
    )

    canvas.restoreState()


# ==================================================
# GENERATE PDF REPORT
# ==================================================

def generate_pdf_report():

    metadata = load_metadata()

    styles = get_styles()

    accuracy = safe_float(metadata.get("accuracy"))
    cv_f1 = safe_float(metadata.get("cv_macro_f1"))
    test_f1 = safe_float(metadata.get("test_macro_f1"))

    elements = []

    # TITLE

    elements.append(
        Paragraph(
            "STUDENT CLASSIFICATION SYSTEM",
            styles["ReportTitle"]
        )
    )

    elements.append(
        Paragraph(
            "MACHINE LEARNING MODEL PERFORMANCE REPORT",
            styles["Heading2"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            f"<b>Report Generated:</b> "
            f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 15))

    # MODEL INFORMATION

    elements.append(
        Paragraph("1. Model Information", styles["SectionHeading"])
    )

    model_data = [
        ["Parameter", "Value"],
        ["Best Model", str(metadata.get("best_model", "N/A"))],
        ["Target Variable", str(metadata.get("target", "N/A"))],
        ["Dataset Records", str(metadata.get("records", "N/A"))],
        ["Number of Features", str(metadata.get("feature_count", "N/A"))],
    ]

    elements.append(
        format_table(model_data, [2.2 * inch, 4.3 * inch])
    )

    elements.append(Spacer(1, 15))

    # OVERALL METRICS

    elements.append(
        Paragraph("2. Overall Performance", styles["SectionHeading"])
    )

    metric_data = [
        ["Evaluation Metric", "Score", "Percentage"],
        ["Test Accuracy", f"{accuracy:.4f}", f"{accuracy * 100:.2f}%"],
        ["Cross Validation Macro F1", f"{cv_f1:.4f}", f"{cv_f1 * 100:.2f}%"],
        ["Test Macro F1", f"{test_f1:.4f}", f"{test_f1 * 100:.2f}%"],
    ]

    elements.append(
        format_table(
            metric_data,
            [2.7 * inch, 1.7 * inch, 2.1 * inch]
        )
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            "Accuracy measures the proportion of correct predictions. "
            "Precision measures the correctness of positive predictions. "
            "Recall measures how many actual positive cases were identified. "
            "F1-score combines precision and recall.",
            styles["SmallText"]
        )
    )

    # CLASSIFICATION REPORT

    elements.append(
        Paragraph(
            "3. Class-wise Precision, Recall and F1 Score",
            styles["SectionHeading"]
        )
    )

    classification_data = get_classification_report(metadata)

    if len(classification_data) > 1:

        elements.append(
            format_table(
                classification_data,
                [
                    1.5 * inch,
                    1.2 * inch,
                    1.2 * inch,
                    1.2 * inch,
                    1.0 * inch
                ]
            )
        )

    else:

        elements.append(
            Paragraph(
                "Classification report is unavailable. "
                "Retrain the model with classification_report saved "
                "in model_metadata.json.",
                styles["SmallText"]
            )
        )

    # CONFUSION MATRIX

    elements.append(
        Paragraph("4. Confusion Matrix", styles["SectionHeading"])
    )

    confusion_data = get_confusion_matrix(metadata)

    if confusion_data:

        elements.append(
            format_table(confusion_data)
        )

    else:

        elements.append(
            Paragraph(
                "Confusion matrix is unavailable in the model metadata.",
                styles["SmallText"]
            )
        )

    # INTERPRETATION

    elements.append(
        Paragraph(
            "5. Evaluation Notes",
            styles["SectionHeading"]
        )
    )

    elements.append(
        Paragraph(
            "The metrics in this report are taken from the saved "
            "model evaluation metadata. Test metrics represent the "
            "held-out test set used during model training. "
            "Cross-validation results represent the saved cross-validation "
            "score. Results should be interpreted with the dataset size, "
            "class distribution and evaluation methodology in mind.",
            styles["SmallText"]
        )
    )

    # FINAL DOCUMENT

    document = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.85 * inch,
        title="Student Model Evaluation Report",
        author="Student Classification System"
    )

    document.build(
        elements,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    print("\nPDF REPORT GENERATED SUCCESSFULLY")
    print("--------------------------------------")
    print("Test Accuracy:", f"{accuracy * 100:.2f}%")
    print("Test Macro F1:", f"{test_f1 * 100:.2f}%")
    print("Cross Validation F1:", f"{cv_f1 * 100:.2f}%")
    print("\nPDF Location:")
    print(PDF_FILE)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    try:
        generate_pdf_report()

    except Exception as error:

        print("\nPDF GENERATION FAILED")
        print("Reason:", error)