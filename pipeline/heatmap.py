from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Starting heatmap generation...")

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/CAM 2.mp4")

heatmap_points = []
frame_count=0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame_count += 1

    if frame_count % 100 == 0:
        print("Processed", frame_count, "frames")

    if frame_count > 500:
        break


    results = model.track(
        frame,
        persist=True,  classes=[0],
        verbose=False
    )

    if results[0].boxes.xyxy is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            heatmap_points.append((center_x, center_y))

cap.release()

print("Collected points:", len(heatmap_points))

if len(heatmap_points) > 0:
    heatmap_points = np.array(heatmap_points)

    plt.figure(figsize=(10, 6))

    plt.hist2d(
        heatmap_points[:, 0],
        heatmap_points[:, 1],
        bins=50
    )
    plt.colorbar()
    plt.title("Store Movement Heatmap")

    plt.savefig("data/heatmap.png")

    print("Heatmap saved to data/heatmap.png")
else:
    print("No points collected")
