# TGR2026 Edge AI Track 2 · Day 3 (TGR-17)

ระบบ instance segmentation เมล็ดกาแฟรายเม็ด 7 คลาส ใช้ YOLO26s-seg (ClassBoost) ที่ export เป็น NCNN FP16 เครื่องเป้าหมายคือ Raspberry Pi 5 รันบน CPU ตอน inference เรียกไลบรารี `ncnn` ของ Python ตรง ๆ ไม่ต้องใช้ PyTorch หรือ Ultralytics มี `client.py` สำหรับดึงภาพจาก server แข่ง ส่งผล polygon ทาง HTTP และส่งสรุปผลขึ้นกระดานคะแนนทาง MQTT ตาม `API_DOCS.md`

## ไฟล์ในโปรเจกต์

| ไฟล์ | หน้าที่ |
|---|---|
| `infer.py` | ประมวลผลภาพทั้งโฟลเดอร์ด้วยโมเดล NCNN เขียน `seg_mask/`, `vis/`, `mqtt/`, `predictions.json` |
| `client.py` | pipeline ของการแข่ง: GET ภาพ → ประมวลผลด้วยขั้นตอนของ `infer.py` → POST ผลทาง HTTP → publish สรุปทาง MQTT |
| `mqtt_sub.py` | subscribe topic MQTT ของทีมเพื่อดูและตรวจ payload ที่ส่งไป |
| `inferpt.py` | แบบเดียวกับ `infer.py` แต่รันจาก checkpoint `.pt` ด้วย PyTorch ใช้เทียบผลกับ NCNN |
| `train.py` | notebook Colab (export เป็น `.py`) สำหรับแพ็กโมเดลที่เทรนแล้วหรือเทรน ClassBoost ใหม่ ฝังชุดโค้ดเทรน `s_classboost_train/` ไว้ข้างใน (ดูหัวข้อ "โมเดลและการเทรน") |
| `requirements.txt` | แพ็กเกจของ `infer.py`, `client.py`, `mqtt_sub.py` |
| `requirements-pt.txt` | แพ็กเกจของ `inferpt.py` (ต้องติดตั้งแยก venv) |
| `API_DOCS.md` | spec ของ HTTP API และ MQTT จากผู้จัด |
| `weights/` | โมเดลที่ใช้ส่งแข่ง (ค่าเริ่มต้นของทุกสคริปต์) |
| `weights_*/` | โมเดลรุ่นอื่นที่เก็บไว้เทียบ (ดูหัวข้อ weights) |

## ติดตั้ง

ใช้ Python virtual environment เปิด Terminal ที่โฟลเดอร์ซึ่งมี `infer.py`

Linux / Raspberry Pi OS 64 บิต:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --timeout 120 -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` มี `numpy`, `Pillow`, `opencv-python-headless`, `ncnn` สำหรับ inference และ `requests`, `paho-mqtt>=2,<3` สำหรับ `client.py` กับ `mqtt_sub.py`

`inferpt.py` ใช้ `requirements-pt.txt` (`torch`, `ultralytics==8.4.215` และอื่น ๆ) ให้ติดตั้งใน venv แยก เพราะ Ultralytics ต้องใช้ `opencv-python` ซึ่งชนกับ `opencv-python-headless` (ทั้งคู่เป็นโมดูล `cv2`)

## ทำนายจากโฟลเดอร์ภาพ (`infer.py`)

```bash
python infer.py --input test/ --weights weights/ --out outputs/
```

| Argument | ความหมาย |
|---|---|
| `--input` | โฟลเดอร์ภาพ อ่านเฉพาะไฟล์ชั้นบนสุด รองรับ `.jpg`, `.jpeg`, `.png` โดยไม่สนตัวพิมพ์เล็กใหญ่ เรียงตามชื่อไฟล์ก่อนประมวลผล ชื่อไฟล์ที่ตัดนามสกุลแล้ว (stem) ต้องไม่ซ้ำกัน |
| `--weights` | โฟลเดอร์โมเดล (ดูหัวข้อ weights) |
| `--out` | โฟลเดอร์ผลลัพธ์ ต้องไม่ใช่โฟลเดอร์ input และไม่อยู่ข้างใน ทุกครั้งที่รันจะลบ `seg_mask/`, `vis/`, `mqtt/`, `predictions.json` ของรอบก่อนทิ้งก่อน (หลังตรวจ input และโหลดโมเดลผ่านแล้ว) ไฟล์อื่นในโฟลเดอร์ไม่ถูกแตะ |
| `--keep` | ไม่ลบผลรอบก่อน ไฟล์ชื่อเดิมถูกเขียนทับ ไฟล์อื่นยังอยู่ |
| `--roi auto\|full` | การตัดบริเวณเม็ดก่อนเข้าโมเดล ค่าเริ่มต้นอ่านจาก `inference_config.json` ใช้ `full` เมื่อพื้นหลังไม่ใช่กระดาษสีอ่อน |
| `--imgsz`, `--conf` | ใช้แทนค่าใน `inference_config.json` |
| `--no-vis` | ไม่สร้างภาพ `vis/` |
| `--runtime` | พิมพ์เวลาโหลดโมเดล เวลาต่อภาพ (อ่านภาพจนบันทึกไฟล์เสร็จ) และสรุปท้ายรัน ไม่เปลี่ยนผลลัพธ์ |
| `--latency` | พิมพ์เวลาแยกขั้นต่อภาพ: อ่านภาพ + inference = `latency_ms` ที่เขียนใน `mqtt/<stem>.json` ตามด้วยเวลาเขียนไฟล์และเวลารวม พร้อมสรุป mean/min/max ท้ายรัน ไม่เปลี่ยนผลลัพธ์ |

สำเร็จจะจบด้วย exit code 0 กรณีหา input/weights ไม่พบ ไม่มีภาพ อ่านภาพไม่ได้ หรือโมเดลไม่ถูกต้อง จะพิมพ์ `ERROR: ...` ทาง stderr และจบด้วย exit code 1

`infer.py` ไม่ได้ warm-up โมเดลก่อนภาพแรก ภาพแรกจึงช้ากว่าภาพอื่น สรุปของ `--runtime` และ `--latency` มีค่าเฉลี่ยที่ตัดภาพแรกออกให้ด้วย

## ผลลัพธ์

```text
outputs/
├── seg_mask/<stem>.png   # หนึ่งไฟล์ต่อภาพ: PNG uint16 ช่องเดียว ขนาดเท่าภาพหลังใช้ EXIF orientation
├── predictions.json      # UTF-8 หนึ่งรายการต่อภาพ: image_name และ objects [{id, class, polygon}]
├── mqtt/<stem>.json      # JSON บรรทัดเดียวต่อภาพ สำหรับกระดานคะแนน MQTT: image, latency_ms, counts, defect_ratio
└── vis/<stem>.jpg        # ภาพ overlay สีต่อเม็ด สำหรับดูด้วยตาเท่านั้น
```

- `seg_mask`: 0 = พื้นหลัง, 1..N = เม็ดที่ 1..N แต่ละพิกเซลเป็นของเม็ดเดียว ถ้าภาพไม่พบเม็ด PNG เป็น 0 ทั้งภาพ และ `objects` เป็น `[]`
- `id` ใน `predictions.json` ตรงกับค่าของเม็ดนั้นใน PNG เรียงจากคะแนนสูงไปต่ำ `class` เป็นตัวพิมพ์เล็ก หนึ่งใน `black, broken, good, husk, immature, parchment, shell` ตาม `API_DOCS.md`
- `polygon` คือจุด `[x, y]` ทุกจุดบนขอบเม็ด (`CHAIN_APPROX_NONE`) ในพิกัดภาพต้นฉบับ พื้นที่ที่ `cv2.fillPoly` ของ polygon ครอบตรงกับพิกเซลของเม็ดนั้นใน PNG พอดี
- `mqtt/<stem>.json` ตาม `API_DOCS.md` ข้อ 2.1:
  - `counts` นับเม็ดตาม `class` ใน `predictions.json` ครบ 7 คีย์ เรียง `good, shell, black, husk, broken, parchment, immature`
  - `defect_ratio = (ทั้งหมด − good) / ทั้งหมด` ปัดทศนิยม 4 ตำแหน่ง ถ้าไม่พบเม็ดเลยเป็น `0.0`
  - `latency_ms` คือเวลาตั้งแต่อ่านภาพจนได้ polygon (ROI, โมเดล, หลังประมวลผล) เป็นจำนวนเต็ม ms (ปัดเศษทิ้ง) ไม่รวมเวลาเขียนไฟล์

ตัวอย่าง `mqtt/<stem>.json`:

```json
{"image": "test_03.jpg", "latency_ms": 150, "counts": {"good": 10, "shell": 0, "black": 0, "husk": 0, "broken": 1, "parchment": 1, "immature": 0}, "defect_ratio": 0.1667}
```

## ส่งผลขึ้นระบบแข่ง (`client.py`)

รันหนึ่งครั้งได้หนึ่งภาพ:

0. โหลดโมเดลและ `inference_config.json` แล้ว warm-up หนึ่งรอบก่อนขอภาพ เวลาโหลดจึงไม่ถูกนับในเวลาที่ server จับ
1. `GET /api/v1/image` ได้ภาพพร้อม header `X-Image-ID`, `X-Image-Name`, `X-Mode`
2. ประมวลผลด้วยขั้นตอนเดียวกับ `infer.py` ได้ objects ชุดเดียวกับที่ `infer.py` ให้กับไฟล์เดียวกัน
3. `POST /api/v1/submit` ส่ง JSON `{"img_id": ..., "predictions": {"image_name": ..., "objects": [...]}}` ซึ่งคือ record ของภาพนี้ใน `predictions.json` จากนั้นจึงเขียนผลลง `--out` (`seg_mask/`, `vis/`, `mqtt/`, `predictions.json`) และเก็บภาพที่ได้ไว้ใน `downloaded_images/` การเขียนไฟล์ทำหลัง submit เพื่อไม่ให้ถูกนับใน `duration_seconds` ของ server
4. publish ไฟล์ `mqtt/<stem>.json` ขึ้น broker ที่ topic `/event/<MQTT_KEY>/<TRACK>/<EVENT>` (WebSocket TLS พอร์ต 443 path `/mqtt/`, MQTT v5, QoS 1) แล้วรอ broker ตอบรับ

ท้ายรันจะดึง `GET /api/v1/stats` มาแสดงจำนวนภาพที่ขอ จำนวนที่ส่ง และโควตา test ที่เหลือ

### ตั้งค่า

แก้ค่าในบล็อก `CONFIGURATION` ด้านบนของ `client.py` หรือส่งผ่าน option ตอนรัน

| ค่าใน `client.py` | option | ความหมาย |
|---|---|---|
| `DEFAULT_HTTP_URL` | `--http-url` | URL ฐานของ server เช่น `https://edgeai-tesa.c333m.site` ใส่เฉพาะ domain ห้ามมี path หรือ `?mode=` |
| `DEFAULT_HTTP_TOKEN` | `--http-token` | รหัสผ่านเว็บของทีม ใช้เป็น `Authorization: Bearer ...` |
| `DEFAULT_TEAM_ID` | `--team-id` | รหัสทีม เป็น username ของ MQTT |
| `DEFAULT_MQTT_KEY` | `--mqtt-key` | MQTT key ที่สร้างจากหน้าเว็บ (คนละค่ากับรหัสผ่านเว็บ) |
| `DEFAULT_MQTT_HOST` | `--mqtt-host` | host ของ broker (ไม่ต้องมี `https://`) |
| `DEFAULT_TRACK` | `--track` | `Embedded`, `EdgeAI` หรือ `Market` (ตัวพิมพ์มีผล) |
| `DEFAULT_EVENT` | `--event` | event ID ใน topic |
| `DEFAULT_WEIGHTS` | `--weights` | โฟลเดอร์โมเดล ค่าเริ่มต้น `weights/` ข้าง `client.py` |
| `DEFAULT_OUT` | `--out` | โฟลเดอร์ผลลัพธ์ ค่าเริ่มต้น `outputs` (นับจากโฟลเดอร์ที่รันคำสั่ง) |

`client.py` เก็บรหัสผ่านเว็บและ MQTT key ของทีมไว้ในไฟล์ ระวังอย่าเผยแพร่ไฟล์นี้หรือ push ขึ้น repository สาธารณะ

### คำสั่ง

```bash
python client.py                        # หนึ่งภาพในโหมดที่ server เปิดอยู่
python client.py --mode train           # GET /api/v1/image?mode=train
python client.py --mode test            # GET /api/v1/image?mode=test (ใช้โควตา test 1 ครั้งต่อภาพที่ได้)
python client.py --skip-mqtt            # ส่งเฉพาะ HTTP
python client.py --latency              # พิมพ์เวลาแยกขั้นตั้งแต่ GET จนส่งผลเสร็จ
python client.py --weights weights_fp16/   # ใช้โมเดลอื่น
```

| Option | ความหมาย |
|---|---|
| `--mode train\|test` | ใส่ `?mode=` ให้ request ดึงภาพ ถ้าไม่ระบุ server ใช้โหมดที่เปิดอยู่ |
| `--skip-mqtt` | ไม่ publish MQTT (ยังเขียน `mqtt/<stem>.json`) |
| `--backend ncnn\|pt` | `ncnn` (ค่าเริ่มต้น) ใช้โมเดลแบบ `infer.py` ส่วน `pt` ใช้ `.pt` แบบ `inferpt.py` (ต้องติดตั้ง `requirements-pt.txt`) |
| `--device` | อุปกรณ์ของ torch เมื่อใช้ `--backend pt` เช่น `cpu`, `cuda` |
| `--no-vis` | ไม่เขียน `vis/<stem>.jpg` |
| `--keep` | ไม่ล้าง `--out` ก่อนเขียน แต่เพิ่ม record ของภาพนี้ลง `predictions.json` (ถ้ามีชื่อภาพเดิมจะแทนที่) ค่าเริ่มต้นคือล้างผลรอบก่อนทิ้ง `--out` จึงมีเฉพาะภาพล่าสุด รอบที่ล้มก่อนได้ผล เช่น GET ได้ 403 จะไม่ลบของเดิม |
| `--latency` | พิมพ์ตารางเวลา: GET, inference (พร้อมค่า `latency_ms`), HTTP submit, รวม GET → submit, เขียนไฟล์, MQTT และรวม GET → MQTT ตอบรับ |

ผลของการรัน:

- exit code 1 เมื่อโหลดโมเดลไม่ได้, GET ไม่สำเร็จ (เช่น 401 รหัสผ่านผิด, 403 โควตาหมด, 503 server หยุด) หรือ submit ถูกปฏิเสธ (เช่น 404, 408, 409) และจะไม่ส่ง MQTT
- ถ้า publish MQTT ไม่สำเร็จ จะพิมพ์ `[-] MQTT Publish Error: ...` แต่ยังจบด้วย exit code 0 เพราะผล HTTP ส่งไปแล้ว
- เมื่อทดสอบกับ server จำลองบน Windows ให้ใช้ `http://127.0.0.1:8000` แทน `http://localhost:8000` เพราะ `localhost` จะลอง IPv6 (`::1`) ก่อน ถ้า server ฟังเฉพาะ IPv4 จะเสียเวลาราว 2 วินาทีต่อ request

## ดู payload MQTT (`mqtt_sub.py`)

```bash
python mqtt_sub.py              # subscribe /event/<MQTT_KEY>/<TRACK>/<EVENT> ด้วยค่าจาก client.py
python mqtt_sub.py --all        # ทุก track/event ของ key นี้
python mqtt_sub.py --count 1    # หยุดหลังได้ข้อความแรก
```

เปิดไว้ก่อนแล้วจึงรัน `client.py` (broker ไม่เก็บข้อความย้อนหลัง) ทุกข้อความจะถูกตรวจว่ามีฟิลด์ครบ มี `counts` ครบ 7 คีย์ และ `defect_ratio` ถูกต้อง broker ของการแข่ง (`tesa-score.c333m.site`) ให้ทีม login ได้แต่ปฏิเสธการ subscribe (`Not authorized`) สคริปต์นี้จึงใช้ได้กับ broker ทดสอบเท่านั้น

## weights

```text
weights/
├── best_ncnn_model/          # export จาก .pt ด้วย Ultralytics format=ncnn quantize=16 (FP16)
│   ├── model.ncnn.param
│   ├── model.ncnn.bin
│   ├── metadata.yaml         # ชื่อคลาส, stride, imgsz, quantize
│   └── model_ncnn.py         # ไฟล์ตัวอย่างจากการ export (infer.py ไม่ได้ใช้)
└── inference_config.json     # ค่าตั้งของ infer.py และ client.py
```

`--weights` ของ `infer.py` และ `client.py` ชี้ได้สองแบบ (`inferpt.py` และ `client.py --backend pt` ใช้โฟลเดอร์ที่มี `best.pt` หรือไฟล์ `.pt` ตรง ๆ):

- โฟลเดอร์ที่มีโฟลเดอร์ export NCNN อยู่ข้างในเพียงโฟลเดอร์เดียว เช่น `--weights weights/`
- โฟลเดอร์ export ตรง ๆ เช่น `--weights weights/best_ncnn_model/` ใช้เลือกโมเดลเมื่อมีหลายโมเดลในโฟลเดอร์เดียวกัน

`inference_config.json` หาในโฟลเดอร์ที่ระบุก่อน ถ้าไม่มีจะหาข้างโมเดลและโฟลเดอร์ถัดขึ้นไปหนึ่งชั้น ถ้าไม่พบเลยจะใช้ค่าเริ่มต้นของโค้ด (`conf` 0.25) `client.py` พิมพ์ path ของไฟล์โมเดลและ config ที่ใช้จริงตอนโหลด ชื่อคลาสใน `metadata.yaml` ต้องเป็น 7 คลาสตามลำดับ `black, broken, good, husk, immature, parchment, shell` ไม่เช่นนั้นจะหยุดพร้อม error

| คีย์ใน `inference_config.json` | ค่าที่ใช้ | ความหมาย |
|---|---|---|
| `imgsz` | 640 | ขนาดภาพเข้าโมเดล (สี่เหลี่ยมจัตุรัส ปัดขึ้นเป็นพหุคูณของ stride 32) |
| `conf` | 0.35 | คะแนนคลาสขั้นต่ำ |
| `roi` | `auto` | `auto` หรือ `full` |
| `max_det` | 500 | จำนวนเม็ดสูงสุดต่อภาพหลัง NMS |

โฟลเดอร์โมเดลอื่น (ข้อมูลจาก `metadata.yaml` ของแต่ละโฟลเดอร์ ทุกตัวเป็น YOLO26s-seg):

| โฟลเดอร์ | ชุดข้อมูลที่เทรน | ความแม่นยำ | ขนาด `.bin` | `inference_config.json` |
|---|---|---|---|---|
| `weights/` | `classboost_s_v1` | FP16 | 19.9 MB | มี (conf 0.35) |
| `weights_final/` | ไฟล์เดียวกับ `weights/` ทุกไบต์ | FP16 | 19.9 MB | มี (conf 0.35) |
| `weights_fp16/` | `day3_s_focus_v1` (baseline ก่อน classboost) | FP16 | 19.9 MB | ไม่มี |
| `weights_fp32/` | `day3_s_focus_v1` | FP32 | 39.6 MB | ไม่มี |
| `weights_sfinetune/` | `day3_s_focus_v1` | FP32 | 39.6 MB | ไม่มี |
| `weights_prune/` | `day3_s_focus_v1` | FP32 | 35.4 MB | มี (conf 0.35) |
| `weights_old/` | `day3_s_focus_v1` | FP32 (ไม่ quantize) | 39.6 MB | ไม่มี |
| `weightspt/` | `best.pt`, `last.pt` (PyTorch) | – | 52 MB | ไม่มี |

โฟลเดอร์ที่ไม่มี `inference_config.json` จะใช้ค่าเริ่มต้นของโค้ด (`conf` 0.25) ถ้าต้องการเทียบกับ `weights/` ให้ใส่ `--conf 0.35` (สำหรับ `infer.py`) หรือคัดลอก `inference_config.json` ไปไว้ด้วย

## การเตรียมภาพ

1. เปิดภาพด้วย Pillow ปรับทิศตาม EXIF แล้วแปลงเป็น RGB ผลทุกอย่างอ้างอิงพิกัดของภาพนี้
2. ROI (`auto`): ย่อภาพให้ด้านยาวไม่เกิน 1024 แล้วแปลงเป็น HSV
   - ถ้าค่ามัธยฐานของ S > 35 หรือของ V < 140 (ไม่ใช่กระดาษสีอ่อน) ใช้ทั้งภาพ
   - ไม่เช่นนั้นหาพิกเซลที่มีสีหรือมืด (`S > 38 และ V < 247` หรือ `V < 80`) ตัดชิ้นเล็กกว่า 12 พิกเซล และแถบแคบยาวที่ติดขอบภาพ (อัตราส่วนด้าน > 8) ออก
   - ใช้กรอบที่ครอบทุกชิ้นที่เหลือ เผื่อขอบ max(24 พิกเซล, 15% ของด้านยาวของกรอบ)
   - ถ้ากรอบใหญ่กว่า 85% ของภาพ หรือไม่พบชิ้นใดเลย ใช้ทั้งภาพ
3. Letterbox แบบ Ultralytics: ย่อหรือขยายภาพที่ตัดแล้วโดยรักษาสัดส่วนด้วย bilinear ให้พอดี 640 × 640 วางกลางบนพื้นสีเทา (114, 114, 114) หาร 255 แล้วจัดเป็น float32 รูป 3 × 640 × 640

## Inference และหลังประมวลผล

4. รันโมเดลด้วย `ncnn` บน CPU (ปิด Vulkan) ได้ผลสองชุด: detection 43 × 8400 (box 4 ค่า, คะแนน 7 คลาส, mask coefficient 32 ค่า) และ prototype 32 × 160 × 160
5. คัด anchor ที่คะแนนคลาสสูงสุด > `conf` ทำ NMS แยกตามคลาสที่ IoU 0.5 เก็บไม่เกิน `max_det` แล้วแปลง box กลับพิกัดภาพที่ตัด
6. Mask: นำ coefficient คูณ prototype ได้ logit ขยายด้วย bilinear เป็นความละเอียดเต็มของภาพที่ตัด เก็บเฉพาะในกรอบ box แล้วใช้ logit > 0 ทิ้งเม็ดที่ mask ว่าง ขั้นที่ 3–6 ทำแบบเดียวกับ `predict(retina_masks=True)` ของ Ultralytics 8.4.215
7. วางเม็ดลง mask จากคะแนนสูงไปต่ำ เม็ดหลังได้เฉพาะพิกเซลที่ยังว่าง เก็บชิ้นต่อเนื่องที่ใหญ่ที่สุด (8-connectivity) เติมรูภายใน แล้วสร้าง polygon ด้วย `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)`
8. ทิ้งเม็ดที่ไม่เหลือพิกเซล เม็ดที่การเติมรูจะทับเม็ดอื่น หรือเม็ดที่เล็กกว่า 16 พิกเซล ทุกการตัดหรือเติมจะรายงานทาง stderr ขึ้นต้นด้วย `POSTPROCESS:` ไม่ทิ้งเงียบ ๆ
9. เลื่อนพิกัดกลับตามตำแหน่ง ROI ให้เป็นพิกัดภาพต้นฉบับ นับ counts และ defect_ratio แล้วบันทึก PNG, JSON, `mqtt/` และ vis

ไม่มี test-time augmentation ไม่ดาวน์โหลดไฟล์ระหว่าง inference และไม่ปรับโมเดลจากภาพ test

## โมเดลและการเทรน

### โมเดลใน `weights/`

- YOLO26s-seg ขนาด S (head `Segment26`, `end2end: false`) 7 คลาส Ultralytics 8.4.215 ไม่ได้ทำ pruning, KD หรือ INT8
- ClassBoost คือการปรับข้อมูลแล้ว fine-tune ต่อจาก baseline ซึ่งเป็นโมเดล S ที่ fine-tune ไว้แล้วบน `day3_s_focus_v1` (baseline คือ `weights_fp16/`) เป็นรุ่นก่อนทำ regularization และก่อน final refit
- export เป็น NCNN FP16 (`quantize=16`) ที่ imgsz 640 เมื่อ 2026-11-19 17:36 UTC ตามข้อมูลใน `weights/best_ncnn_model/metadata.yaml`

### ข้อมูลฝึก

- เก็บชุด train เดิมไว้ทั้งหมด แล้วเพิ่มภาพ copy-paste 200 ภาพ, real augment 8 ชุดต่อภาพ และ hard-source bonus 4 (`hard_probability` 0.75, seed 43)
- คัดภาพยากจากชุด train เท่านั้น ไม่มี pseudo-label เน้นคู่คลาสที่สับสน: broken–shell, good–immature, parchment–immature, black–husk
- ไม่นำภาพ validation (`new_val`, `old_val`) มาสร้างชุด train

### การฝึก

- ฝึกบน Colab NVIDIA GPU สองรอบเริ่มจาก baseline เดียวกัน: `trial_a` ใช้ `cls=0.50` และ `trial_b` ใช้ `cls=0.75`
- AdamW `lr0` 0.00003, `lrf` 0.1, warmup 1 epoch, imgsz 640, batch 2, สูงสุด 20 epochs ต่อรอบ, patience 7, seed 42, deterministic

### การเลือก checkpoint

ประเมิน `best` และ `last` ของทั้งสองรอบเทียบกับ baseline วัดด้วย F1 แยกคลาสที่ mask IoU ≥ 0.95 แล้วเฉลี่ย `new_val` กับ `old_val` เท่ากัน จะเปลี่ยนจาก baseline ได้เมื่อผ่านทุกข้อ:

- F1 ของ `new_val` ไม่ลดลง
- F1 ของ `old_val` และคะแนน geometry ลดได้ไม่เกิน 2 จุดเปอร์เซ็นต์
- คะแนนเฉลี่ยดีขึ้น

| โมเดล | เฉลี่ย | `new_val` | `old_val` |
|---|---|---|---|
| baseline | 0.673 | 0.581 | 0.765 |
| trial_a_best | 0.741 | 0.719 | 0.763 |
| **trial_a_last (ใช้งาน)** | **0.752** | **0.738** | **0.765** |
| trial_b_best | 0.749 | 0.719 | 0.778 |
| trial_b_last | 0.712 | 0.656 | 0.769 |

ตัวเลขมาจาก `provenance.json` ของชุดโมเดลที่ export หลัง export เป็น NCNN FP16 ได้ตรวจซ้ำเทียบกับ baseline FP16 และคำนวณ native Mask mAP ด้วย

ชุด validation นี้ถูกใช้เลือกโมเดลเอง และ `new_val` มีเพียง 4 ภาพ (31 เม็ด) ตัวเลขข้างบนจึงไม่ใช่ผลจากชุด test อิสระ และไม่รับประกันว่าโมเดลไม่ overfit F1 วัดหลัง ROI และการทำความสะอาด mask ที่ conf 0.35 แบบเดียวกับ `infer.py` ส่วน native Mask mAP วัดด้วย validator ของ Ultralytics ที่ conf 0.001

### ใช้ `train.py`

`train.py` เป็น notebook Colab ที่ export เป็น `.py` (ใช้ `google.colab` และ mount Google Drive) ต้องรันบน Colab ทีละเซลล์ตามลำดับ ใช้รันในเครื่องตรง ๆ ไม่ได้

- **เซลล์ 1** แตกชุดโค้ดที่ฝังไว้ (ตรวจ SHA-256) ไปที่ `/content/coffee_classboost_submission_code/s_classboost_train/` แล้วรัน `train.py inspect`
- **เซลล์ 3** เลือก `MODE` มีสองแบบ
  - `package_existing` (ค่าเริ่มต้น): แพ็กโมเดลที่เทรนไว้แล้วจากโครงการ `Day3_ClassBoost/classboost_s_v1` บน Drive ไม่ต้องใช้ GPU ไม่เทรนหรือ export ใหม่ ตรวจว่า checkpoint ที่เลือกเป็น `trial_a_last` แล้วสร้าง ZIP ที่มี `weights/best.pt`, `weights/best_ncnn_model/`, `weights/inference_config.json`, `train.py` พร้อม helper, `provenance.json` และ `checksums.json`
  - `train_classboost`: เทรนใหม่ (เซลล์ 5–10) ต้องใช้ Colab GPU, ตั้งชื่อโครงการใหม่ที่ไม่ใช่ `classboost_s_v1` และต้องมี full-session backup ของ `Day3_Finetune` ที่มีข้อมูลและ label ผลอาจไม่ตรงกับรุ่นเดิมทุกหลัก แม้ใช้ค่าเดียวกัน
- **เซลล์ 11–12** เก็บ ZIP บน Drive และดาวน์โหลด

ชุดโค้ดที่ฝังไว้ (`s_classboost_train/train.py`) เป็น CLI ที่มีคำสั่ง `inspect`, `prepare`, `train`, `evaluate`, `export`, `bundle-existing` และ `all` ค่าฝึกอยู่ใน `configs/classboost.json` แพ็กเกจที่ใช้เทรนอยู่ใน `requirements-train.txt` (`ultralytics==8.4.215`, `ncnn==1.0.20261020`, `pnnx==20260526` ส่วน torch ใช้ตัวที่มากับ Colab GPU)

ข้อมูลฝึกและ checkpoint `.pt` ของโมเดลใน `weights/` ไม่ได้อยู่ใน repository นี้ (`weightspt/` เป็น checkpoint รุ่นก่อน) `.pt` ตัวที่ตรงกันอยู่ใน ZIP ที่ได้จากโหมด `package_existing` ถ้าจะใช้ `inferpt.py` เทียบผลกับ `weights/` ต้องใช้ `.pt` ตัวนั้น
