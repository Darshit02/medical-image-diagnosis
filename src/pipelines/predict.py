import torch
from torchvision import transforms
from PIL import Image

from src.models.model import PneumoniaCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("artifacts/pneumonia_model.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


image_path = "test_xray.jpeg"   # put your xray image here

image = Image.open(image_path).convert("RGB")

image = transform(image).unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs,1)


classes = ["NORMAL","PNEUMONIA"]

print("Prediction:", classes[predicted.item()])