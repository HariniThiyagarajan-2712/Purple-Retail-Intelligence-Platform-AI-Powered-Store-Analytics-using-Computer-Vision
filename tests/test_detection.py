from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/CAM 2.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0]
    )

    annotated_frame = results[0].plot()

    # Current people count
    count = 0
    if results[0].boxes.id is not None:
        count = len(results[0].boxes.id)

    # Draw count on video
    cv2.putText(
        annotated_frame,
        f"People Count: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
