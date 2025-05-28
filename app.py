# นำเข้าไลบรารีที่จำเป็น
import streamlit as st  # ใช้สร้าง UI บนเว็บแอป
import torch  # ไลบรารีสำหรับจัดการกับ Tensor และการประมวลผลโมเดล
import torchvision.transforms as transforms  # สำหรับการแปลงภาพให้อยู่ในรูปที่โมเดลเข้าใจได้
from PIL import Image  # ใช้เปิดและจัดการรูปภาพ
import torch.nn as nn  # สำหรับสร้างเลเยอร์ในโมเดล
from torchvision import models  # ดึงโมเดลสำเร็จรูปมาใช้
import os  # ใช้จัดการไฟล์/โฟลเดอร์ (ในกรณีจำเป็น)

# ระบุ path ของไฟล์โมเดลที่ train มาแล้ว
model_path = 'model/drink_classification.pth'

# เลือกใช้ GPU ถ้ามี ถ้าไม่มีก็ใช้ CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โหลดโมเดล ResNet18 ที่ไม่มี pretrained weights (weights=None)
model = models.resnet18(weights=None)

# ปรับให้เลเยอร์สุดท้าย (Fully Connected) รองรับ 3 คลาสแทนของเดิม
num_ftrs = model.fc.in_features  # จำนวน features ที่เข้าไปใน fc layer เดิม
model.fc = nn.Linear(num_ftrs, 3)  # เปลี่ยนให้รองรับ 3 class: coke, pepsi, sprite

# โหลด weights ที่เทรนไว้แล้วเข้าโมเดล
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)  # ย้ายโมเดลไปยังอุปกรณ์ที่กำหนด (GPU/CPU)
model.eval()  # ตั้งค่าโมเดลให้อยู่ในโหมดประเมินผล (ไม่ใช่โหมดฝึก)

# รายชื่อคลาสที่โมเดลสามารถจำแนกได้
class_names = ['coke', 'pepsi', 'sprite']

# สร้างหน้าตา UI ด้วย Streamlit
st.title("🧃 แยกชนิดเครื่องดื่ม")  # หัวข้อหลัก
st.write("อัปโหลดรูปภาพเครื่องดื่ม (โค้ก / เป๊ปซี่ / สไปรท์)")  # คำอธิบาย

# ส่วนสำหรับให้อัปโหลดรูปภาพ
uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png"])

# เมื่อมีการอัปโหลดไฟล์
if uploaded_file is not None:
    # เปิดภาพที่อัปโหลด และแปลงให้เป็น RGB เพื่อให้แน่ใจว่าเข้ากันได้กับโมเดล
    image = Image.open(uploaded_file).convert('RGB')

    # แสดงภาพบนหน้าเว็บ
    st.image(image, caption='รูปที่อัปโหลด', use_container_width=True)
    st.markdown("🔎 **กำลังประมวลผล...**")  # แจ้งสถานะ

    # เตรียมภาพให้พร้อมใช้งานกับโมเดล (Preprocessing)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ปรับขนาดให้เป็น 224x224 พิกเซล (ขนาดที่โมเดลรับได้)
        transforms.ToTensor(),  # แปลงภาพให้เป็น Tensor
        transforms.Normalize(  # ปรับค่าพิกเซลให้มีค่าเฉลี่ยและส่วนเบี่ยงเบนตามมาตรฐานของ ImageNet
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # เพิ่ม batch dimension และย้ายไปยังอุปกรณ์

    # ทำการพยากรณ์โดยไม่ใช้ gradient (ประหยัดหน่วยความจำ)
    with torch.no_grad():
        outputs = model(img_tensor)  # ส่งภาพเข้าโมเดล
        probs = torch.softmax(outputs, dim=1)[0]  # คำนวณความน่าจะเป็นของแต่ละคลาส
        _, predicted = torch.max(outputs, 1)  # หาคลาสที่โมเดลคาดว่ามีความน่าจะเป็นมากที่สุด

    # แปลง index ของคลาสให้เป็นชื่อ
    predicted_label = class_names[predicted.item()]
    confidence = probs[predicted.item()].item() * 100  # เปลี่ยนเป็นเปอร์เซ็นต์

    # แสดงผลลัพธ์บนหน้าเว็บ
    st.success(f"✅ คาดว่าเป็น **{predicted_label.upper()}** ({confidence:.2f}%)")