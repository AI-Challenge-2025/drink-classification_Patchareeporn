# 🍹 Drink Classification (แยกประเภทเครื่องดื่ม: โค้ก / เป๊ปซี่ / สไปรท์)
แอป Streamlit สำหรับจำแนกภาพเครื่องดื่มออกเป็น 3 ประเภท ได้แก่:
- โค้ก
- เป๊ปซี่
- สไปรท์
โมเดลถูกฝึกด้วย BDH X-Brain และสามารถใช้งานผ่านหน้าเว็บได้ทันที โดยการอัปโหลดภาพ

## 🧠 โมเดลที่ใช้
- ประเภท: Image Classification (Classification)
- เครื่องมือ: [BDH X-Brain](https://xbrain.bdh.ai/)
- โมเดลที่เทรนแล้วถูกนำมาใช้ในแอปนี้ผ่าน UUID ที่กำหนด

## 🚀 วิธีใช้งาน
### 1. Clone โปรเจกต์
git clone https://github.com/yourusername/drink-classification-app.git
cd drink-classification-app
### 2. สร้าง Virtual Environment และติดตั้งไลบรารี
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
### 3. รันแอป
streamlit run app.py
### 4. เปิดเบราว์เซอร์ที่:
 Local URL: http://localhost:8501
 Network URL: http://192.168.1.191:8501

## 🖼 วิธีใช้งาน
1. เปิดแอปด้วย Streamlit
2. เลือกอัปโหลดภาพเครื่องดื่ม
3. ระบบจะแสดงผลลัพธ์ว่าเป็นเครื่องดื่มประเภทใด (โค้ก / เป๊ปซี่ / สไปรท์)

## 📂 โครงสร้างโปรเจกต์
drink-classification-app/
├── app.py               # ไฟล์หลักของ Streamlit
├── requirements.txt     # รายการไลบรารีที่ต้องติดตั้ง
└── README.md            # รายงานโปรเจกต์

## 🧪 ตัวอย่างภาพที่ใช้เทรน
- เก็บข้อมูลภาพจาก Google / เว็บไซต์ที่ไม่มีลิขสิทธิ์
- รูปภาพ 30 รูปต่อคลาส

## ✨ เหมาะสำหรับ
- โครงงานด้าน Machine Learning เบื้องต้น
- ตู้ขายอัตโนมัติ / ระบบจัดหมวดหมู่สินค้า
- ตัวอย่างการใช้งานโมเดลภาพผ่าน Web UI

## 🙋‍♀️ ผู้พัฒนา
- ชื่อ: นางสาวพัชรีพร พฤฒิสาร
- โครงงานนี้เป็นส่วนหนึ่งของวิชา [AI/ระบบสมองกลฝังตัวและอิเล็กทรอนิกส์สื่อสาร]
- ส่งวันที่ 28 พ.ค. 2568
