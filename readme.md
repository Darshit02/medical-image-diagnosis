# 🩺 AI Radiology Assistant — Pneumonia Detection from Chest X-Ray

An AI-powered medical imaging system that detects **Pneumonia from Chest X-ray images** using a **Convolutional Neural Network (CNN)** built with **PyTorch**.

The system includes:

* Deep learning model for medical image classification
* FastAPI backend for predictions
* Streamlit dashboard UI for interaction
* Grad-CAM heatmap for explainable AI
* Downloadable radiology report

---

# 🚀 Features

✔ Chest X-ray pneumonia detection
✔ Deep learning CNN model (PyTorch)
✔ Grad-CAM heatmap visualization
✔ Radiology-style Streamlit dashboard
✔ FastAPI prediction API
✔ Interactive X-ray vs heatmap comparison
✔ Downloadable AI diagnosis report

---

# 🧠 Model Architecture

The model is a **Convolutional Neural Network (CNN)** trained on chest X-ray images.

Architecture overview:

Input Image (224x224x3)

↓
Conv Layer → ReLU → MaxPool

↓
Conv Layer → ReLU → MaxPool

↓
Conv Layer → ReLU → MaxPool

↓
Fully Connected Layer

↓
Output (NORMAL / PNEUMONIA)

---

# 📊 Dataset

Dataset used: **Chest X-Ray Pneumonia Dataset**

Classes:

* NORMAL
* PNEUMONIA

Dataset structure:

data/
chest_xray/

train/
NORMAL/
PNEUMONIA/

test/
NORMAL/
PNEUMONIA/

val/
NORMAL/
PNEUMONIA/

---

# 🧾 Project Structure

medical-image-diagnosis/

artifacts/
pneumonia_model.pth
gradcam_result.jpg

app/
api.py
streamlit_app.py

src/

models/
model.py

pipelines/
train.py
predict.py

utils/
data_loader.py
gradcam.py
report_generator.py

requirements.txt
README.md

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/medical-image-diagnosis.git
cd medical-image-diagnosis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🏋️ Train the Model

```bash
python -m src.pipelines.train
```

This will train the CNN and save the model in:

artifacts/pneumonia_model.pth

---

# 🔬 Run the Prediction API

Start the FastAPI backend:

```bash
uvicorn app.api:app --reload
```

API will run at:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

---

# 💻 Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

The web interface will open automatically.

---

# 🫁 Explainable AI (Grad-CAM)

The system uses **Grad-CAM** to visualize which lung regions influenced the model's prediction.

Features:

* Lung region attention heatmap
* Interactive comparison slider
* Radiology interpretation panel

---

# 📄 AI Radiology Report

After analysis the system generates a **downloadable PDF report** including:

* Patient information
* AI diagnosis
* Model confidence
* Radiology interpretation

---

# 🧪 Example Workflow

1. Upload chest X-ray
2. Run AI diagnosis
3. View prediction and confidence
4. Explor
