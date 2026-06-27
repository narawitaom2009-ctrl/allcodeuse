from ultralytics import YOLO

model = YOLO(r"#path best.pt")

results = model.predict(
    source=r"#pathimage.jpg",
    conf=0.25,
    save=True
)

results[0].save(filename="output.jpg")
print("บันทึกภาพแล้ว: output.jpg")
# Debug: เช็คว่ามี box ไหม
result = results[0]
print("จำนวน box ที่ตรวจพบ:", len(result.boxes))