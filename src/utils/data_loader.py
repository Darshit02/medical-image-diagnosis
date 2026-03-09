import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    root="data/raw/chest_xray/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="data/raw/chest_xray/val",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="data/raw/chest_xray/test",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)       

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))
print("Classes:", train_dataset.classes)


# Test bench 
images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

plt.imshow(images[0].permute(1,2,0))
plt.title(f"Label : {labels[0].item()}")
plt.axis("off")
plt.show()