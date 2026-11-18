# สเปกส่งงาน Day 2 — Mask และ Polygon

## คำสั่งมาตรฐาน

```sh
python infer.py --input test/ --weights weights/ --out outputs/
```

ใช้ชื่อ argument เดียวกับ Day 1: --input, --weights, --out ส่วนในสไลด์ที่เขียน --weight/--output ให้ใช้ชื่อตามคำสั่งมาตรฐานนี้

--weights รับโฟลเดอร์ ไม่บังคับชื่อ best.pt หรือนามสกุลไฟล์ มีหลายโมเดลและ config ได้ infer.py จัดการโหลดเอง ระบุวิธีใน README รันบน Raspberry Pi 4 ด้วย CPU ไม่เปิด GUI ไม่ดาวน์โหลดโมเดลขณะ inference ไม่ฝึกหรือปรับโมเดลจาก test และไม่ทำ random augmentation/TTA

## โครงสร้าง ZIP และผลลัพธ์

```text
team_XX_day2/
├── README.md
├── requirements.txt
├── train.py
├── infer.py
├── weights/
│   └── ...
└── outputs/
    ├── seg_mask/
    │   ├── img_01.png
    │   └── img_02.png
    ├── predictions.json
    └── vis/
        └── ...
```

requirements.txt สำหรับ inference เท่านั้น README อธิบายวิธีติดตั้ง preprocessing และ weights ที่ใช้ vis/ เก็บภาพ overlay mask เพื่อให้กรรมการดูด้วยตา ไม่ใช้แทน mask จริง ไม่ต้องสร้าง timings.csv ผู้จัดจับเวลาเอง

## seg_mask/: mask รายเม็ด

- รวม mask ทุกภาพไว้ในโฟลเดอร์เดียว ไม่แยกโฟลเดอร์คลาส
- หนึ่งภาพ test ต้องมีหนึ่ง PNG ใช้ชื่อเดิมเปลี่ยนนามสกุล เช่น img_01.jpg → img_01.png
- เป็นภาพ integer channel เดียว แนะนำ uint16 PNG ไม่ใช่ RGB หรือภาพสี overlay
- ขนาดเท่าภาพต้นฉบับหลังใช้ EXIF orientation ค่า 0 = พื้นหลัง; ค่า 1..N = เม็ดที่ 1..N
- แต่ละเม็ดมี ID ของตัวเอง แต่ละพิกเซลอยู่ได้หนึ่ง instance ไม่มีการส่ง bbox แทน mask
- ถ้า resize/pad ภาพเข้าโมเดล ต้องแปลง mask กลับพิกัดภาพต้นฉบับก่อนส่ง
- ภาพที่ไม่พบเม็ด: PNG มีค่า 0 ทั้งภาพ และ objects เป็น []

## predictions.json

UTF-8 เป็น array มีหนึ่งรายการต่อภาพ input ตามตัวอย่างนี้ ไม่ใช้ RLE และไม่ใช้โครงสร้าง COCO ของสเปกฉบับก่อน

```json
[
  {
    "image_name": "img_01.jpg",
    "objects": [
      {
        "id": 1,
        "class": "broken",
        "polygon": [[412, 230], [431, 226], [450, 238], [447, 262], [425, 260]]
      },
      {
        "id": 2,
        "class": "broken",
        "polygon": [[518, 301], [540, 297], [552, 315], [536, 330], [520, 322]]
      }
    ]
  },
  {"image_name": "img_02.jpg", "objects": []}
]
```

| Field | ความหมาย |
|---|---|
| image_name | ชื่อไฟล์ภาพ test พร้อมนามสกุล ไม่ใช่ absolute path |
| objects | รายการเม็ดในภาพ หนึ่งเม็ดหนึ่ง object |
| id | จำนวนเต็ม 1..N ตรงกับค่าของเม็ดนั้นใน seg_mask PNG เริ่มใหม่ได้ทุกภาพ |
| class | หนึ่งใน black, broken, good, husk, immature, parchment, shell ใช้ตัวพิมพ์เล็กทั้งหมด |
| polygon | array ของจุด [x,y] เรียงตามขอบเม็ด หน่วยพิกเซลของภาพต้นฉบับ อย่างน้อย 3 จุด |

ไม่มี bbox, RLE หรือ score ในรูปแบบนี้ ใช้ชื่อคลาสตัวพิมพ์เล็ก รวมถึง parchment ให้ตรงกับ Day 1

สร้าง polygon จาก mask ของเม็ดเดียวกันด้วย cv2.findContours และเก็บจุดขอบให้ครบ เช่น CHAIN_APPROX_NONE ไม่ลดจำนวนจุดจนขอบหาย ต้องตรงกับ seg_mask จุดอยู่ในขอบเขตภาพ โดย x เป็นคอลัมน์ y เป็นแถว เริ่มจาก 0 ซ้ายบน ไม่จำเป็นต้องใส่จุดแรกซ้ำท้ายรายการ

หาก mask มีหลายส่วนแยกกันหรือรูภายใน รูปแบบ polygon เดียวนี้ไม่สามารถแทนทุกส่วนได้ครบ ต้องตรวจให้ mask ที่ส่งสอดคล้องกับ polygon; ไม่ทิ้งส่วนของ mask อย่างเงียบ ๆ PNG เป็นข้อมูล mask ระดับพิกเซลสำหรับการเทียบ

## พฤติกรรมและการตรวจ

- รองรับ .jpg/.jpeg/.png ไม่สนตัวพิมพ์เล็กใหญ่ เรียงชื่อไฟล์ก่อนประมวลผล
- ชื่อไฟล์และ stem ของภาพ input ต้องไม่ซ้ำกัน เพื่อไม่ให้ PNG เขียนทับกัน
- ต้องมี JSON และ PNG ครบทุกภาพ ไม่แก้ไข ย้าย หรือลบ input และเก็บ output แยกจาก input
- สำเร็จ exit code 0; input/weights ไม่พบ ไม่มีภาพ อ่านภาพไม่ได้ หรือผลผิดรูปแบบ ให้แสดง error และจบด้วย nonzero code ไม่ข้ามภาพเงียบ ๆ
- จับคู่ภาพด้วย image_name จับคู่ object.id กับ instance ID ใน PNG ไม่ใช้ลำดับ object ทำนายไปเทียบกับลำดับ ground truth โดยตรง
- ตามสไลด์ ตรวจด้วย F1 โดยใช้ mask IoU threshold 0.95: จับคู่ mask ทำนายกับเฉลยแบบหนึ่งต่อหนึ่ง คู่ที่ IoU >= 0.95 เป็น TP; ที่เหลือเป็น FP/FN; F1 = 2TP/(2TP+FP+FN)
- สไลด์ที่แนบไม่ได้ระบุชัดว่าการจับคู่ต้องบังคับ class ตรงกันด้วยหรือไม่ ผู้จัดต้องประกาศเรื่องนี้ก่อนแข่งขัน

## ไฟล์แจกและเฉลย

แจก DAY2_INFER_SPEC.md และ day2_json/submission_example.json ให้ผู้เข้าแข่งขัน
เก็บ day2_json/ground_truth.json และ day2_json/seg_mask/ สำหรับผู้ตรวจเท่านั้น ไม่แจกเฉลย
เฉลยนี้ใช้ mask ที่ผู้ใช้ตรวจแล้วทั้ง 7 ภาพ และ class เดิมของแต่ละเม็ด ไม่เปลี่ยน label เดิม
