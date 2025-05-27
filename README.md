1. ติดตั้ง dependencies
```bash
pip install -r requirements.txt

# 🧃 Drink Classification

โปรเจกต์นี้เป็นระบบแยกชนิดเครื่องดื่ม (Coke / Pepsi / Sprite) โดยใช้โมเดล **ResNet18** จาก PyTorch และแสดงผลผ่านเว็บแอปด้วย **Streamlit**

---

## 📂 โครงสร้างโปรเจกต์
Drink-Classification/
├── app.py # เว็บแอปสำหรับอัปโหลดและจำแนกภาพเครื่องดื่ม
├── train.py # โค้ดสำหรับฝึกโมเดล
├── drink_classification.pth # ไฟล์โมเดลที่ผ่านการฝึกแล้ว
├── dataset/ # โฟลเดอร์ภาพ training (แบ่งตามโฟลเดอร์ย่อย coke/pepsi/sprite)
├── requirements.txt # รายการไลบรารีที่ใช้
└── README.md # ไฟล์อธิบายโปรเจกต์นี้

---

## ▶️ วิธีใช้งาน

### 1. ติดตั้งไลบรารีที่จำเป็น
```bash
pip install -r requirements.txt
2. ฝึกโมเดล (เฉพาะกรณีต้องการ train ใหม่)
เตรียม dataset ไว้ในโฟลเดอร์ dataset/ แยกเป็นโฟลเดอร์ย่อย coke/, pepsi/, sprite/ ภายใน

จากนั้นรันคำสั่ง:

bash
Copy
Edit
python train.py
หลังจากฝึกเสร็จ จะได้ไฟล์ชื่อ drink_classification.pth สำหรับใช้ในเว็บแอป

3. เปิดเว็บแอป
bash
Copy
Edit
streamlit run app.py
เปิดเบราว์เซอร์ไปที่ http://localhost:8501/

อัปโหลดรูปเครื่องดื่ม และระบบจะแสดงผลว่าเป็น Coke / Pepsi / Sprite พร้อมเปอร์เซ็นต์ความมั่นใจ

🧠 โมเดลที่ใช้
ใช้โครงสร้าง ResNet18 จาก torchvision.models โดยปรับ layer สุดท้ายให้รองรับ 3 คลาส

เทรนด้วย CrossEntropyLoss และ Adam Optimizer

รูปภาพถูก Resize และ Normalize ตามค่ามาตรฐานของ ImageNet

✨ ความสามารถของระบบ
รองรับการพยากรณ์ภาพจากไฟล์ .jpg, .jpeg, .png

ใช้งานง่ายผ่านเว็บแอป (ไม่ต้องติดตั้ง GUI เพิ่มเติม)

สามารถเทรนใหม่ได้ง่ายด้วย train.py

📝 ข้อเสนอแนะเพิ่มเติม
สามารถเพิ่มคลาสใหม่ เช่น Fanta, Est ฯลฯ ได้โดยเพิ่มโฟลเดอร์ใหม่ใน dataset/ แล้วปรับค่าที่เกี่ยวข้องในโค้ด

ปรับแต่ง UI ของ Streamlit ให้มีความสวยงามยิ่งขึ้น

เพิ่มระบบ Drag & Drop หรือแสดงผลแบบ Batch หลายภาพพร้อมกัน 

##กรณีที่ bdh-ai-api.botnoi.ai ใช้งานได้ 
import streamlit as st
import requests  # สำหรับส่ง HTTP request ไป API
from PIL import Image  # จัดการรูปภาพ
import io  # สำหรับแปลงรูปภาพเป็น bytes

# ตั้งค่าหน้าเว็บแอป
st.set_page_config(page_title="Drink Classifier", layout="centered")
st.title("🧃 แยกชนิดเครื่องดื่ม")

# ให้ผู้ใช้เลือกอัปโหลดไฟล์ภาพประเภท jpg/jpeg/png
uploaded_file = st.file_uploader("อัปโหลดรูปภาพเครื่องดื่ม (โค้ก / เป๊ปซี่ / สไปรท์)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # เปิดภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    # แสดงภาพบนเว็บ
    st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)

    # แจ้งสถานะกำลังประมวลผล
    st.write("🔎 กำลังประมวลผล...")

    # แปลงภาพเป็น bytes เพื่อส่งผ่าน HTTP
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')  # แปลงภาพเป็น PNG แล้วบันทึกลง buffer
    image_bytes = image_bytes.getvalue()  # ดึงข้อมูล byte ออกมา

    # กำหนด URL ของ API ที่จะส่งข้อมูลไปพยากรณ์
    url = "https://bdh-ai-api.botnoi.ai/v1/prediction/94dbeaa3-4b20-48b0-b604-7a4f4d68ee52/predict"
    files = {"file": ("image.png", image_bytes, "image/png")}  # สร้าง payload สำหรับไฟล์ภาพ

    # ส่ง HTTP POST request ไป API พร้อมไฟล์ภาพ
    response = requests.post(url, files=files)

    if response.status_code == 200:
        # ถ้าการตอบกลับสำเร็จ
        result = response.json()  # แปลงผลลัพธ์ JSON เป็น dict
        pred = result['prediction']  # ดึงค่าการพยากรณ์ออกมา
        st.success(f"✅ คาดว่าเป็น: **{pred}**")  # แสดงผลลัพธ์บนเว็บ
    else:
        # กรณีเกิดข้อผิดพลาด เช่น API ล่ม หรือเชื่อมต่อไม่ได้
        st.error("❌ มีปัญหาในการเชื่อมต่อ API หรือโมเดล")

