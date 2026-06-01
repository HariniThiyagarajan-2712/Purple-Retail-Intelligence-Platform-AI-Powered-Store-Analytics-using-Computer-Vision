from ultralytics import YOLO
import cv2
import json

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/CAM 2.mp4")

seen_ids = set()
events = []
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame_count += 1

    if frame_count > 300:
        break

    results = model.track(frame, persist=True, classes=[0],verbose=False)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        for pid in ids:
            if pid not in seen_ids:
                seen_ids.add(pid)

                events.append({
                    "event": "NEW_VISITOR",
                    "person_id": int(pid),
                    "camera": "CAM2"
                })

cap.release()

with open("data/events.json", "w") as f:
    json.dump(events, f, indent=4)

print("Events Saved:", len(events))
print("Seen IDs:", seen_ids)
print("Total Unique IDs:", len(seen_ids))
