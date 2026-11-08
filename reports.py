import io
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet


def csv_report(records):

    return pd.DataFrame(records).to_csv(
        index=False
    ).encode("utf-8")


def pdf_report(records):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4)
    )

    data = [[
        "Student ID",
        "Name",
        "Department",
        "Prediction",
        "Faculty"
    ]]

    for record in records:

        data.append([
            record.student_id,
            record.name,
            record.department,
            record.prediction,
            record.created_by
        ])

    table = Table(data, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8)
    ]))

    styles = getSampleStyleSheet()

    document.build([
        Paragraph(
            "Student Classification Report",
            styles["Title"]
        ),
        Spacer(1, 15),
        table
    ])

    return buffer.getvalue()