# นำเข้าไลบรารีที่เกี่ยวข้องกับการฝึกสอนโมเดล
import torch  # ไลบรารีหลักของ PyTorch
import torch.nn as nn  # สำหรับสร้างเลเยอร์ของโมเดล และ loss function
from torchvision import datasets, transforms, models  # ใช้โหลด dataset และโมเดลสำเร็จรูป
from torch.utils.data import DataLoader  # สำหรับจัดการข้อมูลที่ใช้ในการฝึก

# กำหนด path ของ dataset ที่จัดเตรียมไว้ในโฟลเดอร์ชื่อ 'dataset'
data_dir = 'dataset'

# สร้างชุดของ transform เพื่อปรับภาพให้เหมาะกับการฝึกสอน
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ปรับขนาดภาพให้เท่ากัน (224x224)
    transforms.ToTensor(),  # แปลงภาพให้เป็น Tensor (ค่าพิกเซล 0-1)
    transforms.Normalize(  # ปรับค่าพิกเซลให้มีค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน (ตาม ImageNet)
        mean=[0.485, 0.456, 0.406],  # ค่าเฉลี่ย RGB
        std=[0.229, 0.224, 0.225]  # ค่าส่วนเบี่ยงเบนมาตรฐาน RGB
    )
])

# โหลดภาพจากโฟลเดอร์ โดยโฟลเดอร์ย่อยจะใช้เป็นชื่อคลาส (เช่น 'coke', 'pepsi', 'sprite')
dataset = datasets.ImageFolder(data_dir, transform=transform)
print("Classes:", dataset.classes)  # แสดงชื่อคลาสที่พบใน dataset

# สร้าง DataLoader สำหรับโหลดข้อมูลเป็นชุดๆ
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)  # โหลดทีละ 16 รูป แบบสุ่ม

# สร้างโมเดล ResNet18 (ไม่ใช้ weights ที่ pre-trained)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # ใช้ GPU ถ้ามี
model = models.resnet18(weights=None)  # โหลดโมเดล ResNet18 แบบเปล่า (ไม่มีการ pretrain)

# ปรับเลเยอร์ fully connected (fc) สุดท้าย ให้รองรับ 3 คลาส
num_ftrs = model.fc.in_features  # จำนวน input ของ fc เดิม
model.fc = nn.Linear(num_ftrs, 3)  # เปลี่ยน output ให้มี 3 คลาส
model = model.to(device)  # ย้ายโมเดลไปยัง GPU หรือ CPU

# กำหนด loss function และ optimizer
criterion = nn.CrossEntropyLoss()  # ใช้ cross entropy สำหรับ classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # ใช้ Adam optimizer ในการปรับ weights

# ตั้งค่ารอบการฝึก (epoch)
model.train()  # ตั้งโมเดลให้อยู่ในโหมดฝึก (เปิด dropout/batchnorm)
num_epochs = 20  # ฝึกทั้งหมด 20 รอบ
for epoch in range(num_epochs):
    running_loss = 0.0  # ใช้เก็บค่าความสูญเสีย (loss) ของรอบนั้น ๆ

    for inputs, labels in train_loader:  # วนลูปแต่ละ batch
        inputs, labels = inputs.to(device), labels.to(device)  # ย้ายข้อมูลไปยังอุปกรณ์ (GPU/CPU)

        optimizer.zero_grad()  # เคลียร์ gradient เดิม
        outputs = model(inputs)  # ส่งข้อมูลเข้าโมเดล
        loss = criterion(outputs, labels)  # คำนวณ loss ระหว่าง prediction กับ label จริง
        loss.backward()  # คำนวณ gradient ย้อนกลับ
        optimizer.step()  # ปรับ weights ตาม gradient ที่ได้

        running_loss += loss.item()  # รวมค่า loss ของ batch นั้น

    # แสดงค่า loss ของแต่ละ epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

# เมื่อฝึกเสร็จแล้ว บันทึก weights ของโมเดลไว้ในไฟล์ .pth
torch.save(model.state_dict(), 'drink_classification.pth')
print("Training finished, model saved as drink_classification.pth")
