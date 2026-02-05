# FaceBack SimSnap

<img width="616" height="616" alt="Image"
src="https://github.com/user-attachments/assets/f6d7cbf1-3b1b-4e91-ae16-6c9dbd7d059d"
/>

---

## โครงสร้างไฟล์และการทำงานที่สำคัญ
```text
FACEBACK_SIMSNAP/
│
├── backend/                       # ส่วนประมวลผล Backend
│   ├── static/
│   │   └── fonts/                # โฟลเดอร์เก็บฟอนต์ Sarabun สำหรับการแสดงผลภาษาไทย
│   ├── analysis_engine.py        # ไฟล์ประมวลผลหลักสำหรับคำนวณและผลลัพธ์
│   ├── main.py                   # ไฟล์หลักสำหรับรัน Server และจัดการ API Endpoints
│   └── requirements.txt          # Library Python ที่จำเป็น
│
├── frontend/                      # ส่วนแสดงผล Frontend
│   ├── public/                   # ไฟล์ Static และไฟล์ index.html หลัก
│   │   └── cover/
│   │       └── cover.png          # ภาพปกของหน้าเว็บ
│   ├── src/
│   │   ├── components/            # ส่วนประกอบหน้าจอ
│   │   │   ├── Home.js            # หน้าหลัก
│   │   │   ├── ModelSelector.js   # หน้าเลือกโมเดล / นำเข้าข้อมูล
│   │   │   ├── Processing.js      # หน้าแสดงสถานะการประมวลผล
│   │   │   └── Results.js         # หน้าแสดงผลลัพธ์
│   │   ├── App.js                 # ควบคุมการแสดงหน้าเว็บ
│   │   └── index.js               # จุดเริ่มต้นการทำงานของ React
│   ├── tailwind.config.js         # การตั้งค่า Tailwind CSS (Font Sarabun)
│   └── package.json               # การตั้งค่า Library และ Scripts ของ Frontend
│
└── README.md                      # คู่มือการติดตั้งและใช้งานระบบ
```

## ความต้องการของระบบ

### สำหรับเครื่อง Server
- **OS:** Windows 11 (64-bit)
- **CPU:** 64-bit ความเร็วขั้นต่ำ 2.5 GHz (4 Core ขึ้นไป)
- **RAM:** ขั้นต่ำ 16 GB (แนะนำ 32 GB)
- **Disk Space:** ว่างไม่น้อยกว่า 10 GB

### สำหรับ Software
- **Python:** เวอร์ชัน 3.11.x  
- **Node.js:** เวอร์ชัน 20.x (LTS)

---

## ขั้นตอนการติดตั้ง

### ส่วนที่ 1 การติดตั้ง Server (Backend)

1. Clone Repository
   ```bash
   git clone https://github.com/w1t1ta/FaceBack_SimSnap.git

2. เข้าสู่โฟลเดอร์ Backend
    ```bash
    cd FaceBack_SimSnap/backend

3. สร้าง Virtual Environment (Python 3.11)
    ```bash
    py -3.11 -m venv venv

4. เปิดใช้งาน Virtual Environment
    ```bash
    venv\Scripts\activate
( เมื่อสำเร็จจะมี (venv) แสดงหน้าบรรทัดคำสั่ง )

5. ติดตั้งไลบรารีที่จำเป็น
    ```bash
    pip install -r requirements.txt
    
### ส่วนที่ 2 การติดตั้ง Client (Frontend)

1. เปิด Command Prompt ใหม่ และเข้าสู่โฟลเดอร์ frontend
    ```bash
    cd FaceBack_SimSnap/frontend
    
2. ติดตั้งไลบรารี dependencies
    ```bash
    npm install

## การเริ่มใช้งานระบบ

### ขั้นตอนที่ 1 รัน Back-end

1. เปิด Command Prompt ที่อยู่ในโฟลเดอร์ backend และ activate venv แล้ว
2. รันคำสั่ง
    ```bash
    uvicorn main:app

### ขั้นตอนที่ 2 รัน Front-end

1. เปิด Command Prompt ที่อยู่ในโฟลเดอร์ frontend
2. รันคำสั่ง
    ```bash
    npm start
3. ระบบจะเปิด Web Browser อัตโนมัติที่ http://localhost:3000
