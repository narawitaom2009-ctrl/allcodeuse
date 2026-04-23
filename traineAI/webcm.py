import cv2
from ultralytics import YOLO

# โหลดโมเดล
model = YOLO(r"C:\Users\ASUS\vs\testcatdog\runs\detect\train-5\weights\best.pt")

# เปิดกล้อง (0 = webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ใช้ YOLO detect
    results = model(frame)

    # วาดผลลัพธ์ลงบนภาพ
    annotated_frame = results[0].plot()

    # แสดงผล
    cv2.imshow("YOLO Camera", annotated_frame)

    # กด ESC เพื่อออก
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()