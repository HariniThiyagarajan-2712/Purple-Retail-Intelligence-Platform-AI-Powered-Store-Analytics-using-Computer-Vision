from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/CAM 2.mp4")

unique_ids = set()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0]
    )

    if results[0].boxes.id is not None:
      ids = results[0].boxes.id.cpu().numpy().astype(int)

        for pid in ids:
            unique_ids.add(pid)

        print(
            f"Current People: {len(ids)} | "
            f"Unique Visitors: {len(unique_ids)}"
        )

cap.release()

print("Total Unique Visitors:", len(unique_ids))
