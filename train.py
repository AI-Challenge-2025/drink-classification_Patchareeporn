import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# กำหนด path ของ dataset
data_dir = 'dataset'

# สร้าง transform สำหรับข้อมูล
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# โหลด dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
print("Classes:", dataset.classes)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

# สร้างโฟลเดอร์ model ถ้ายังไม่มี
os.makedirs('model', exist_ok=True)

# บันทึกโมเดลลงในโฟลเดอร์ model/
torch.save(model.state_dict(), 'model/drink_classification.pth')
print("Training finished, model saved as model/drink_classification.pth")  # แก้เป็น # แทน //
