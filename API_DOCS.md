# 📖 เอกสารคู่มือ API การแข่งขัน TopGun (คู่มือสำหรับผู้เข้าแข่งขัน)

เอกสารนี้ระบุรายละเอียดข้อกำหนดการเชื่อมต่อ API ทั้งฝั่ง **HTTP Server (ส่งผลพิกัด Ground Truth เพื่อตรวจคะแนน)** และ **MQTT Broker (ส่งผลสรุปขึ้น Scoreboard)** สำหรับผู้เข้าแข่งขัน

---

## ⚠️ คำเตือนสำคัญมาก: ความแตกต่างของรหัสผ่าน (Credentials)

| บริการ | ข้อมูลยืนยันตัวตน (Key / Password) | ที่มาของ Key | วิธีการใช้งาน |
| :--- | :--- | :--- | :--- |
| **HTTP Evaluation Server** | **HTTP Bearer Token** | คือ **Password ที่ทีมใช้ล็อกอินเข้าสู่ระบบเว็บไซต์** | ใส่ใน Header: `Authorization: Bearer <TEAM_PASSWORD>` |
| **MQTT Broker** | **MQTT Key** | **ต้องกด Generate บนเว็บไซต์ก่อนใช้งาน** (ในหน้า Dashboard / My Team) | ใช้เป็น Password เชื่อมต่อ MQTT และใส่ใน Topic |

> 🚫 **ห้ามสับสน:** ห้ามนำ Password ล็อกอินเว็บไซต์ไปต่อ MQTT และห้ามนำ MQTT Key ไปยิงใส่ HTTP Server

---

## 🔄 ลำดับขั้นตอนการทำงานของทีม (Pipeline Workflow)

ในแต่ละรอบของการทดสอบ โปรแกรมของแต่ละทีมจะต้องทำตาม 4 ขั้นตอนนี้เรียงตามลำดับ:

```
[1. GET Image via HTTP] ──► ได้รับไฟล์ภาพ + X-Image-ID + X-Image-Name
          │
          ▼
[2. AI Model Inference] ──► ประมวลผลตรวจจับเมล็ดกาแฟ (Polygon & Counts)
          │
          ├─────────────────────────────────────────┐
          ▼                                         ▼
[3. POST Submit via HTTP]                 [4. Publish via MQTT]
 ส่งพิกัด Polygon ตรวจความแม่นยำ           ส่งผลสรุป Counts & Defect Ratio
 (เข้าสู่เซิร์ฟเวอร์ประเมินผล)                        (ขึ้นระบบ TESA )
```

---

## 1. HTTP Server Endpoints (เซิร์ฟเวอร์ประเมินผล)

* **Server URL:** ตามที่ผู้ดูแลระบบประกาศ (เช่น `https://api.yourdomain.com` หรือ `http://<IP>:8000`)
* **Header ยืนยันตัวตน:**
  ```http
  Authorization: Bearer YOUR_TEAM_PASSWORD
  ```

---

### 1.1 ดึงภาพทดสอบ: `GET /api/v1/image`

ดึงไฟล์ภาพมาประมวลผล (สามารถระบุ `?mode=train` หรือ `?mode=test` ได้ หากไม่ระบุระบบจะอิงตามโหมดที่เปิดอยู่ขณะนั้น)

* **Method:** `GET`
* **URL:** `/api/v1/image`
* **Request Headers:**
  ```http
  Authorization: Bearer YOUR_TEAM_PASSWORD
  ```
* **Response Headers สำคัญที่ระบบส่งกลับมา:**
  * **`X-Image-ID`:** รหัสเฉพาะของรอบทดสอบนี้ (เช่น `img_6b683e369e7a4da1`) **จำเป็นต้องเก็บไว้ใช้ในขั้นตอนที่ 3**
  * **`X-Image-Name`:** ชื่อไฟล์รูปภาพจริง (เช่น `test_03.jpg` หรือ `Parchment_test.jpg`) **จำเป็นต้องนำชื่อนี้ไปใส่ใน JSON**
  * **`X-Mode`:** โหมดปัจจุบัน (`train` หรือ `test`)
* **Response Body:** ไบนารีของรูปภาพ (`image/jpeg` หรือ `image/png`)

---

### 1.2 ส่งผลการทำนายพิกัด: `POST /api/v1/submit`

ส่งผลลัพธ์พิกัด Polygon และคลาสของเมล็ดกาแฟที่ตรวจจับได้กลับมาบันทึกผล

* **Method:** `POST`
* **URL:** `/api/v1/submit`
* **Request Headers:**
  ```http
  Authorization: Bearer YOUR_TEAM_PASSWORD
  Content-Type: application/json
  ```
* **Request Body JSON Format:**
  ```json
  {
    "img_id": "img_6b683e369e7a4da1",
    "predictions": {
      "image_name": "test_03.jpg",
      "objects": [
        {
          "id": 1,
          "class": "parchment",
          "polygon": [
            [767, 1335],
            [766, 1336],
            [765, 1336],
            [764, 1336]
          ]
        },
        {
          "id": 2,
          "class": "broken",
          "polygon": [
            [500, 800],
            [520, 810],
            [510, 830]
          ]
        }
      ]
    }
  }
  ```
* **Response (`200 OK`):**
  ```json
  {
    "status": "success",
    "message": "Submission recorded successfully",
    "img_id": "img_6b683e369e7a4da1",
    "image_name": "test_03.jpg",
    "mode": "train",
    "duration_seconds": 0.428,
    "received_at": "2026-11-19T03:22:38.874317+00:00"
  }
  ```
  *(ค่า `duration_seconds` คือเวลาที่เซิร์ฟเวอร์จับเวลานับตั้งแต่ส่งภาพเสร็จสมบูรณ์จนกระทั่งได้รับ JSON กลับมา)*

---

### 1.3 ตรวจสอบสถิติและโควตาของทีม: `GET /api/v1/stats`

* **Method:** `GET`
* **URL:** `/api/v1/stats`
* **Request Headers:**
  ```http
  Authorization: Bearer YOUR_TEAM_PASSWORD
  ```
* **Response (`200 OK`):**
  ```json
  {
    "team_name": "TEAM_01",
    "train_mode": {
      "images_requested": 15,
      "submissions_received": 15
    },
    "test_mode": {
      "quota_limit": 5,
      "images_requested": 2,
      "submissions_received": 2,
      "remaining_attempts": 3
    }
  }
  ```

---

## 2. MQTT Broker (ส่งข้อมูลสรุปขึ้นกระดานคะแนน)

* **Broker Host:** `tesa-score.c333m.site`
* **Port:** `443` (WebSocket TLS / Secure WSS)
* **WebSocket Path:** `/mqtt/`
* **Username:** รหัสทีมของคุณ เช่น `TEAM_01`
* **Password:** **MQTT Key ที่สร้างจากหน้าเว็บ**
* **Topic:** `/event/<YOUR_MQTT_KEY>/<TRACK>/<EVENT>`
  * ตัวอย่าง: `/event/c8f1e240a1b2c3d4/EdgeAI/1`
  * ค่า `TRACK` ที่เลือกได้ (Case-sensitive): `Embedded`, `EdgeAI`, หรือ `Market`
  * ค่า `EVENT`: ปกติคือ `1`

### 2.1 โครงสร้างข้อมูลที่ต้องส่งผ่าน MQTT (Payload Format)

ข้อมูลที่ส่งผ่าน MQTT จะเป็น JSON บรรทัดเดียว มีโครงสร้างดังนี้:

```json
{
  "image": "test_03.jpg",
  "latency_ms": 150,
  "counts": {
    "good": 10,
    "shell": 0,
    "black": 0,
    "husk": 0,
    "broken": 1,
    "parchment": 1,
    "immature": 0
  },
  "defect_ratio": 0.1667
}
```

#### รายละเอียดฟิลด์ของ MQTT:
1. **`image`** *(string)*: ชื่อไฟล์ภาพ ต้องตรงกับชื่อภาพที่ได้จาก Header `X-Image-Name` ในขั้นตอนดึงภาพ
2. **`latency_ms`** *(integer)*: ระยะเวลาที่ใช้ประมวลผลโมเดลในเครื่องของคุณ (หน่วยมิลลิวินาที ms)
3. **`counts`** *(object)*: จำนวนเมล็ดกาแฟที่นับได้แยกตามแต่ละคลาส:
   * `good`: เมล็ดสมบูรณ์
   * `shell`: เมล็ดรูปหอย
   * `black`: เมล็ดดำ
   * `husk`: เมล็ดติดเปลือก
   * `broken`: เมล็ดแตกหัก
   * `parchment`: เมล็ดกะลา
   * `immature`: เมล็ดอ่อน
4. **`defect_ratio`** *(float)*: อัตราส่วนเมล็ดเสีย คำนวณจาก:
   $$\text{defect\_ratio} = \frac{\text{ผลรวมของ defect ทั้งหมด}}{\text{จำนวนเมล็ดทั้งหมด}}$$
   *(หากไม่พบเมล็ดใดๆ ให้ใส่ `0.0`)*

---

## 3. ตัวอย่างโปรแกรมเชื่อมต่อครบทุกขั้นตอน (`client_test.py`)

ติดตั้งไลบรารีที่จำเป็นก่อนเริ่มรัน:
```bash
pip install requests "paho-mqtt>=2,<3"
```

โครงสร้างโค้ดแบบย่อ (ดูไฟล์เต็มได้ที่ [client_test.py](file:///e:/ai/topgun_server/client_test.py)):

```python
import requests
import json
import ssl
import paho.mqtt.client as mqtt

# -------------------------------------------------------------
# กำหนดค่าเชื่อมต่อของทีม
# -------------------------------------------------------------
HTTP_URL = "http://127.0.0.1:8000"         # URL ของ HTTP Server
HTTP_TOKEN = "YOUR_TEAM_PASSWORD"          # Password เว็บไซต์ของทีม
MQTT_HOST = "tesa-score.c333m.site"
TEAM_ID = "TEAM_01"
MQTT_KEY = "YOUR_GENERATED_MQTT_KEY"       # Key ที่กดสร้างจากหน้าเว็บ
TRACK = "EdgeAI"
EVENT = "1"

# 1. ดึงภาพผ่าน HTTP
headers = {"Authorization": f"Bearer {HTTP_TOKEN}"}
resp = requests.get(f"{HTTP_URL}/api/v1/image", headers=headers)
img_id = resp.headers["X-Image-ID"]
img_name = resp.headers["X-Image-Name"]
image_bytes = resp.content

# 2. ประมวลผลโมเดล (สมมุติผลลัพธ์)
predictions = {
    "image_name": img_name,
    "objects": [{"id": 1, "class": "parchment", "polygon": [[767, 1335], [766, 1336]]}]
}
mqtt_data = {
    "image": img_name,
    "latency_ms": 140,
    "counts": {"good": 10, "shell": 0, "black": 0, "husk": 0, "broken": 0, "parchment": 1, "immature": 0},
    "defect_ratio": 0.0909
}

# 3. ส่งผลลัพธ์พิกัดกลับทาง HTTP
requests.post(f"{HTTP_URL}/api/v1/submit", json={"img_id": img_id, "predictions": predictions}, headers=headers)

# 4. ส่งผลสรุปสถิติผ่าน MQTT WebSocket
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="websockets", protocol=mqtt.MQTTv5)
client.username_pw_set(TEAM_ID, MQTT_KEY)
client.ws_set_options(path="/mqtt/")
client.tls_set_context(ssl.create_default_context())
client.connect(MQTT_HOST, 443)
client.loop_start()
client.publish(f"/event/{MQTT_KEY}/{TRACK}/{EVENT}", json.dumps(mqtt_data).encode("utf-8"), qos=1)
client.disconnect()
```

---

## 4. รหัสแจ้งสถานะข้อผิดพลาด (HTTP Status Codes)

| Status Code | ความหมาย | สาเหตุและวิธีแก้ไข |
| :--- | :--- | :--- |
| `200 OK` | สำเร็จ | ทำรายการสำเร็จ |
| `401 Unauthorized` | เข้าสู่ระบบไม่ผ่าน | Password ของทีมใน `Authorization: Bearer` ไม่ถูกต้อง |
| `403 Forbidden` | โควตาเต็ม หรือโหมดปิดอยู่ | ในโหมด Test คุณใช้โควตาครบ $N$ ครั้งแล้ว หรือระบบยังไม่เปิดให้ทดสอบโหมดนี้ |
| `408 Request Timeout` | ส่งผลช้าเกินเวลา | ส่งผล JSON ช้ากว่าเวลาที่ผู้คุมสอบกำหนด |
| `409 Conflict` | ส่งผลซ้ำ | รูปภาพรหัส `img_id` นี้ได้เคยส่งผลตรวจไปแล้ว (ห้ามส่งภาพเดิมซ้ำ) |
| `503 Service Unavailable`| เซิร์ฟเวอร์หยุดชั่วคราว | กรรมการทำการกดหยุดระบบชั่วคราว |
