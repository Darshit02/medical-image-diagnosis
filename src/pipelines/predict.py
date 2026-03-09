import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

from src.models.model import PneumoniaCNN
from src.utils.gradcam import GradCAM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("artifacts/pneumonia_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image_path = "test_xray.jpeg"

image = Image.open(image_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

gradcam = GradCAM(model, model.conv3)

heatmap = gradcam.generate(input_tensor)

image_np = np.array(image.resize((224,224)))

heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

cv2.imwrite("artifacts/gradcam_result.jpg", overlay)

print("Grad-CAM saved to artifacts/gradcam_result.jpg")