import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# กำหนด device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# โหลดโมเดลเหมือนตอนฝึก
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model.load_state_dict(torch.load('model/drink_classification.pth', map_location=device))
model = model.to(device)
model.eval()  # เปลี่ยนเป็นโหมดประเมินผล

# ชื่อคลาส (เรียงตาม label index)
classes = ['coke', 'pepsi', 'sprite']

# ฟังก์ชันสำหรับเตรียมภาพ
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # เพิ่มมิติ batch

# ฟังก์ชันทำนายภาพ
def predict(image_path):
    image_tensor = transform_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    label = predict(image_path)
    print(f"Prediction: {label}")
