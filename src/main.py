import base64
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from src.models.model import PneumoniaCNN
from src.utils.gradcam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("artifacts/pneumonia_model.pth", map_location=device))
model.eval()

gradcam = GradCAM(model, model.conv3)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["NORMAL", "PNEUMONIA"]

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file.file.seek(0)
    pil_image = Image.open(file.file).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    result = classes[predicted.item()]
    confidence = float(outputs.max().item())

    heatmap_base64 = None
    try:
        input_for_gradcam = transform(pil_image).unsqueeze(0).to(device)
        input_for_gradcam.requires_grad_(True)
        heatmap = gradcam.generate(input_for_gradcam)
        image_np = np.array(pil_image.resize((224, 224)))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(".jpg", overlay_rgb)
        heatmap_base64 = base64.b64encode(buffer).decode()
    except Exception:
        pass

    return {
        "prediction": result,
        "confidence": confidence,
        "heatmap": heatmap_base64,
    }