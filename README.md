# โปรแกรมตรวจจับใบหน้าด้วย Azure AI

โปรเจกต์นี้เป็นระบบตรวจจับใบหน้าที่ใช้เทคโนโลยี Azure Face API สามารถวิเคราะห์รูปภาพและระบุตำแหน่งใบหน้า พร้อมทั้งข้อมูลรายละเอียดต่างๆ เช่น ท่าทางหัว การบดบัง และอุปกรณ์เสริม

## ✨ คุณสมบัติหลัก

- ✅ ตรวจจับใบหน้าได้หลายใบหน้าในรูปภาพเดียว
- ✅ วิเคราะห์ท่าทางหัว (Yaw, Pitch, Roll)
- ✅ ตรวจสอบการบดบังบริเวณใบหน้า (หน้าผาก, ตา, ปาก)
- ✅ ระบุอุปกรณ์เสริม (แว่นตา, หมวก เป็นต้น)
- ✅ สร้างรูปภาพที่มีกรอบแสดงใบหน้าที่ตรวจพบ
- ✅ สร้างรายงานผลการวิเคราะห์แบบข้อความ
- ✅ รองรับข้อความภาษาไทย

## 📋 ข้อกำหนดเบื้องต้น

### ซอฟต์แวร์ที่จำเป็น
- Python 3.7 หรือสูงกว่า
- บัญชี Microsoft Azure
- Azure Face API service

### Python Libraries ที่ต้องติดตั้ง
```bash
pip install azure-ai-vision-face
pip install python-dotenv
pip install pillow
pip install matplotlib
pip install numpy
```

## ⚙️ การตั้งค่าเริ่มต้น

### 1. สร้าง Azure Face API Service
1. เข้าสู่ [Azure Portal](https://portal.azure.com)
2. สร้าง Face API resource ใหม่
3. คัดลอก Endpoint และ API Key

### 2. ตั้งค่าไฟล์ Environment
สร้างไฟล์ `.env` ในโฟลเดอร์โปรเจกต์:
```env
AI_SERVICE_ENDPOINT=your_azure_endpoint_here
AI_SERVICE_KEY=your_azure_api_key_here
```

### 3. เตรียมรูปภาพ
- สร้างโฟลเดอร์ `images/`
- วางไฟล์รูปภาพที่ต้องการวิเคราะห์

## 🚀 วิธีการใช้งาน

### การใช้งานพื้นฐาน
```bash
python detect-people.py
```
โปรแกรมจะวิเคราะห์ไฟล์ `images/people.jpg` โดยอัตโนมัติ

### ระบุไฟล์รูปภาพเอง
```bash
python detect-people.py path/to/your/image.jpg
```

## 📊 ผลลัพธ์ที่ได้รับ

### 1. รูปภาพที่มีการแสดงผล
- ไฟล์: `detected_faces_{จำนวนใบหน้า}_faces.jpg`
- มีกรอบสีล้อมรอบใบหน้าที่ตรวจพบ
- แสดงหมายเลขใบหน้าแต่ละใบ

### 2. รายงานข้อความ
- ไฟล์: `face_report_{จำนวนใบหน้า}_faces.txt`
- ข้อมูลรายละเอียดของแต่ละใบหน้า
- สถิติสรุปผลการวิเคราะห์

## 📝 ตัวอย่างผลลัพธ์

```
=== รายงานการตรวจจับใบหน้า ===
ไฟล์: images/people.jpg
จำนวนใบหน้า: 2
วันที่วิเคราะห์: 2025-05-29 22:09:55

ใบหน้าที่ 1:
  ตำแหน่ง: (221, 86)
  ขนาด: 124 x 124
  ท่าหัว: Yaw=-5.2°, Pitch=-10.0°, Roll=-8.1°
  การบัง: หน้าผาก=False, ตา=False, ปาก=False

ใบหน้าที่ 2:
  ตำแหน่ง: (498, 166)
  ขนาด: 111 x 111
  ท่าหัว: Yaw=-8.3°, Pitch=-9.2°, Roll=-6.3°
  การบัง: หน้าผาก=False, ตา=False, ปาก=False
  อุปกรณ์เสริม: AccessoryType.GLASSES
```

## 🛠️ การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

#### 1. ไม่พบไฟล์ .env
```
ข้อผิดพลาด: ไม่พบ AI_SERVICE_ENDPOINT หรือ AI_SERVICE_KEY
```
**วิธีแก้:** ตรวจสอบว่าไฟล์ `.env` อยู่ในโฟลเดอร์โปรเจกต์และมีค่าที่ถูกต้อง

#### 2. ไม่พบไฟล์รูปภาพ
```
ข้อผิดพลาด: ไม่พบไฟล์รูปภาพ images/people.jpg
```
**วิธีแก้:** ตรวจสอบพาธไฟล์และสร้างโฟลเดอร์ `images/` หากจำเป็น

#### 3. API Key ไม่ถูกต้อง
```
ข้อผิดพลาด: Invalid API key
```
**วิธีแก้:** ตรวจสอบ API Key และ Endpoint ใน Azure Portal

## 📁 โครงสร้างโปรเจกต์

```
face-detection-ai-use-Azure-Ai-/
├── detect-people.py          # ไฟล์โปรแกรมหลัก
├── .env                      # ไฟล์การตั้งค่า 
├── README.md                 # เอกสารคู่มือ
├── images/                   # โฟลเดอร์เก็บรูปภาพ
│   └── people.jpg           # รูปภาพตัวอย่าง
├── detected_faces_*.jpg      # รูปภาพผลลัพธ์
└── face_report_*.txt        # รายงานผลการวิเคราะห์
```

## 🔒 ความปลอดภัย

- **ห้าม** commit ไฟล์ `.env` เข้า Git repository
- เก็บ API Key ไว้เป็นความลับ
- ใช้ `.gitignore` เพื่อป้องกันไฟล์ sensitive

## 📜 License

โปรเจกต์นี้เป็น Open Source สำหรับการศึกษาและพัฒนา

## 🤝 การสนับสนุน

หากพบปัญหาหรือต้องการความช่วยเหลือ กรุณาสร้าง Issue ในโปรเจกต์นี้

## 📚 เอกสารอ้างอิง

- [Azure Face API Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/face/)
- [Python SDK for Azure Face](https://pypi.org/project/azure-ai-vision-face/)
- [PIL/Pillow Documentation](https://pillow.readthedocs.io/)