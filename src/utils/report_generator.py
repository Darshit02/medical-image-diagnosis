from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime


def generate_report(patient_name, age, gender, result, confidence):

    file_path = "artifacts/diagnosis_report.pdf"

    c = canvas.Canvas(file_path, pagesize=letter)

    c.setFont("Helvetica", 14)

    c.drawString(200, 750, "AI Medical Diagnosis Report")

    c.setFont("Helvetica", 12)

    c.drawString(50, 700, f"Patient Name: {patient_name}")
    c.drawString(50, 680, f"Age: {age}")
    c.drawString(50, 660, f"Gender: {gender}")

    c.drawString(50, 620, f"Diagnosis Result: {result}")
    c.drawString(50, 600, f"Model Confidence: {confidence}%")

    c.drawString(
        50,
        560,
        "Recommendation: Clinical evaluation recommended if symptoms persist."
    )

    date = datetime.datetime.now().strftime("%Y-%m-%d")

    c.drawString(50, 520, f"Generated on: {date}")

    c.save()

    return file_path