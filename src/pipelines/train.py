import torch
import torch.nn as nn
import torch.optim as optim

from src.models.model import PneumoniaCNN
from src.utils.data_loader import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = PneumoniaCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5 
for epoch in range(epochs):
    model.train()
    running_loss = 0
     
    for images,labels in train_loader :
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    
    print(f"Epoch : {epoch+1}/{epochs} , loss: {running_loss:.4f}")
    # accurecy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "pneumonia_model.pth")

    print("Model saved successfully")
