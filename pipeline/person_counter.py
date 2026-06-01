from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/CAM 3.mp4")

total_detections = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
         if cls == 0:  # person class
            count += 1

    total_detections += count

    cv2.putText(
        frame,
        f"People: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Finished")
