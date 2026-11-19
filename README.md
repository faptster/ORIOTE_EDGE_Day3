# TGR2026 Edge AI Track 2 · Day 2

ระบบ instance segmentation เมล็ดกาแฟรายเม็ด ใช้ YOLO11n-seg ที่ export เป็น NCNN รันบน Raspberry Pi 4 ด้วย CPU ตอน inference เรียกไลบรารี `ncnn` ของ Python ตรง ๆ ไม่ต้องใช้ PyTorch หรือ Ultralytics

## ติดตั้ง

แนะนำ Raspberry Pi OS 64 บิต เปิด Terminal ที่โฟลเดอร์ซึ่งมี `infer.py` แล้วติดตั้งด้วย Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --timeout 120 -r requirements.txt
```

`requirements.txt` มีเฉพาะแพ็กเกจสำหรับ inference: `numpy`, `Pillow`, `opencv-python-headless`, `ncnn`

## ทำนาย

```bash
python infer.py --input test/ --weights weights/ --out outputs/
```

| Argument | ความหมาย |
|---|---|
| `--input` | โฟลเดอร์ภาพ อ่านเฉพาะไฟล์ชั้นบนสุด รองรับ `.jpg`, `.jpeg`, `.png` โดยไม่สนตัวพิมพ์เล็กใหญ่ เรียงตามชื่อไฟล์ก่อนประมวลผล ชื่อไฟล์ที่ตัดนามสกุลแล้ว (stem) ต้องไม่ซ้ำกัน |
| `--weights` | โฟลเดอร์โมเดล (ดูหัวข้อ weights) |
| `--out` | โฟลเดอร์ผลลัพธ์ ต้องไม่ใช่โฟลเดอร์ input และไม่อยู่ข้างใน ไฟล์ชื่อเดิมจะถูกเขียนทับ แต่ไฟล์อื่นจากรอบก่อนจะไม่ถูกลบ จึงควรใช้โฟลเดอร์ว่าง |
| `--roi auto\|full` | การตัดบริเวณเม็ดก่อนเข้าโมเดล ค่าเริ่มต้นอ่านจาก `inference_config.json` ใช้ `full` เมื่อพื้นหลังไม่ใช่กระดาษสีอ่อน |
| `--imgsz`, `--conf` | ใช้แทนค่าใน `inference_config.json` |
| `--no-vis` | ไม่สร้างภาพ `vis/` |
| `--runtime` | พิมพ์เวลาโหลดโมเดล เวลาต่อภาพ (อ่านภาพจนบันทึกไฟล์เสร็จ) และสรุปท้ายรัน ไม่เปลี่ยนผลลัพธ์ |

สำเร็จจะจบด้วย exit code 0 กรณีหา input/weights ไม่พบ ไม่มีภาพ อ่านภาพไม่ได้ หรือโมเดลไม่ถูกต้อง จะพิมพ์ `ERROR: ...` ทาง stderr และจบด้วย exit code 1

## ผลลัพธ์

```text
outputs/
├── seg_mask/<stem>.png   # หนึ่งไฟล์ต่อภาพ: PNG uint16 ช่องเดียว ขนาดเท่าภาพหลังใช้ EXIF orientation
├── predictions.json      # UTF-8 หนึ่งรายการต่อภาพ: image_name และ objects [{id, class, polygon}]
└── vis/<stem>.jpg        # ภาพ overlay สีต่อเม็ด สำหรับดูด้วยตาเท่านั้น
```

- `seg_mask`: 0 = พื้นหลัง, 1..N = เม็ดที่ 1..N แต่ละพิกเซลเป็นของเม็ดเดียว ถ้าภาพไม่พบเม็ด PNG เป็น 0 ทั้งภาพ และ `objects` เป็น `[]`
- `id` ใน `predictions.json` ตรงกับค่าของเม็ดนั้นใน PNG เรียงจากคะแนนสูงไปต่ำ `class` เป็นหนึ่งใน `black, broken, good, husk, immature, parchment, shell`
- `polygon` คือจุด `[x, y]` ทุกจุดบนขอบเม็ด (`CHAIN_APPROX_NONE`) ในพิกัดภาพต้นฉบับ พื้นที่ที่ `cv2.fillPoly` ของ polygon ครอบตรงกับพิกเซลของเม็ดนั้นใน PNG พอดี

## weights

```text
weights/
├── best_ncnn_model/          # export จาก best.pt ด้วย Ultralytics format=ncnn
│   ├── model.ncnn.param
│   ├── model.ncnn.bin
│   ├── metadata.yaml         # ชื่อคลาส, stride, imgsz
│   └── model_ncnn.py         # ไฟล์ตัวอย่างจากการ export (infer.py ไม่ได้ใช้)
├── inference_config.json     # ค่าตั้งของ infer.py
└── best.pt                   # checkpoint PyTorch (infer.py ไม่ได้ใช้ เก็บไว้สำหรับ export ใหม่)
```

`infer.py` ใช้โฟลเดอร์ย่อยใน `weights/` ที่มีไฟล์ `.param` และ `.bin` ซึ่งต้องมีเพียงโฟลเดอร์เดียว ชื่อคลาสใน `metadata.yaml` ต้องเป็น 7 คลาสตามลำดับ `black, broken, good, husk, immature, parchment, shell` ไม่เช่นนั้นจะหยุดพร้อม error

| คีย์ใน `inference_config.json` | ค่าที่ใช้ | ความหมาย |
|---|---|---|
| `imgsz` | 640 | ขนาดภาพเข้าโมเดล (สี่เหลี่ยมจัตุรัส ปัดขึ้นเป็นพหุคูณของ stride 32) |
| `conf` | 0.25 | คะแนนคลาสขั้นต่ำ |
| `roi` | `auto` | `auto` หรือ `full` |
| `max_det` | 500 | จำนวนเม็ดสูงสุดต่อภาพหลัง NMS |

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
9. เลื่อนพิกัดกลับตามตำแหน่ง ROI ให้เป็นพิกัดภาพต้นฉบับ แล้วบันทึก PNG, JSON และ vis

ไม่มี test-time augmentation ไม่ดาวน์โหลดไฟล์ระหว่าง inference และไม่ปรับโมเดลจากภาพ test

## เทรน

โมเดลคือ YOLO11n-seg 7 คลาส เทรนด้วย Ultralytics 8.4.215 ค่าด้านล่างอ่านจาก `train_args` ที่บันทึกใน `weights/best.pt`

| ค่า | ที่ใช้ |
|---|---|
| ขนาดภาพ / batch | 640 / 4 |
| epochs | 100 (patience 25, หยุดเมื่อไม่ดีขึ้น ได้ผลดีสุดที่ epoch 14) |
| optimizer | AdamW, lr0 0.000909, lrf 0.01, seed 42, deterministic |
| augmentation | flip ซ้ายขวาและบนล่าง 0.5, translate 0.05, scale 0.15, HSV (0.005, 0.1, 0.15), ไม่ใช้ mosaic/mixup/copy-paste |
| ผลบนชุด val | mask mAP50 0.959, mask mAP50-95 0.891 |

`train.py` เรียก Ultralytics ด้วยค่าตั้งใน `training_config.json` (คีย์ `model` และ `kwargs`) ที่อยู่ข้างไฟล์ `train.py` ต้องติดตั้ง `ultralytics==8.4.215` และ `pyyaml` เพิ่ม (ไม่อยู่ใน `requirements.txt` เพราะใช้เฉพาะตอนเทรน)

```bash
python train.py --data /path/to/dataset/data.yaml --device 0
# เลือกได้: --epochs N --batch N
```

ถ้าโฟลเดอร์ของ `data.yaml` มี `images/train` และ `images/val` สคริปต์จะเขียน `data_local.yaml` ที่ชี้ path ใหม่ให้ ผลเทรนอยู่ที่ `runs/beans_reproduce/` จากนั้น export เป็น NCNN แล้ววางโฟลเดอร์ผลลัพธ์ใน `weights/`

```bash
yolo export model=runs/beans_reproduce/weights/best.pt format=ncnn imgsz=640
```
