import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base64
import io
import streamlit as st
import requests
from PIL import Image
import time
from streamlit_image_comparison import image_comparison
from src.utils.report_generator import generate_report

st.set_page_config(page_title="Radiology AI System", layout="wide")


st.markdown("""

<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

st.markdown("""

<style>

.header{
font-size:34px;
font-weight:700;
color:#0E6BA8;
}

.sub{
color:#555;
font-size:16px;
margin-bottom:10px;
}

.good{
background:#e6f7ed;
padding:15px;
border-radius:8px;
color:#1b6e3c;
font-weight:600;
}

.bad{
background:#fdeaea;
padding:15px;
border-radius:8px;
color:#8a1f1f;
font-weight:600;
}

.analysis{
background:#f8fafc;
color:#334155;
padding:15px;
border-radius:8px;
border-left:4px solid #0E6BA8;
}

</style>

""", unsafe_allow_html=True)


st.markdown("""

<div class="header">
<i class="fa-solid fa-stethoscope"></i>
AI Radiology Chest X-Ray System
</div>
<div class="sub">
Deep Learning Assistant for Pneumonia Detection
</div>
""", unsafe_allow_html=True)

st.divider()

uploaded_file = None
image = None


left, right = st.columns(2)


with left:
    st.markdown("### Patient Information")
    name = st.text_input("Patient Name", "John Doe")
    age = st.slider("Age", 1, 100, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    st.divider()
    st.markdown("### Upload Chest X-ray")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=350)


with right:
    st.markdown("### AI Diagnosis")

    if uploaded_file is not None and image is not None:
        analyze = st.button("Analyze X-ray")
        if analyze:
            with st.spinner("Radiology AI analyzing image..."):
                time.sleep(2)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )
                data = response.json()
                result = data["prediction"]
                # API returns logits; clamp to 0-100 for display
                confidence = max(0, min(100, int(data.get("confidence", 0.5) * 100)))

            st.divider()
            if result == "NORMAL":
                st.markdown(
                    "<div class='good'>NORMAL — No Pneumonia Detected</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='bad'>PNEUMONIA DETECTED</div>",
                    unsafe_allow_html=True)

            st.write("")
            st.write("Model Confidence")
            st.progress(max(0.0, min(1.0, confidence / 100)))
            st.write(f"{confidence}%")
            st.divider()

            st.markdown("### Lung Region Detection")
            heatmap_b64 = data.get("heatmap")
            if heatmap_b64:
                heatmap_bytes = base64.b64decode(heatmap_b64)
                heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
                image_comparison(
                    img1=image,
                    img2=heatmap_image,
                    label1="Original X-ray",
                    label2="AI Heatmap"
                )
            else:
                st.info("Heatmap not generated.")

            st.divider()

            st.markdown("### AI Radiology Interpretation")
            if result == "PNEUMONIA":
                st.markdown("""
<div class="analysis">
Possible pulmonary infection patterns detected.

AI identified abnormal opacity in lung regions that may indicate pneumonia.
Clinical confirmation recommended.

</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div class="analysis">
No abnormal radiographic patterns detected.

Lung structure appears consistent with a normal chest X-ray.

</div>
""", unsafe_allow_html=True)

            st.divider()

            # ---------- REPORT DOWNLOAD ----------
            st.markdown("### Download Radiology Report")
            report_path = generate_report(name, age, gender, result, confidence)
            with open(report_path, "rb") as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name="radiology_report.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("Upload an X-ray image to start analysis.")

st.divider()

st.caption("AI Radiology Assistant • Pneumonia Detection")
