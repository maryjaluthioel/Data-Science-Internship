from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam with DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Give the hardware camera sensor a brief moment to warm up
cv2.waitKey(1000)

if not cap.isOpened():
    print("[ERROR] Could not initialize webcam device.")
    exit()

print("[SYSTEM] Camera initialized. Starting display window...")

while True:
    ret, frame = cap.read()

    # If a single frame drops, don't crash or break the window loop
    if not ret or frame is None:
        cv2.waitKey(1)
        continue

    # Run YOLOv8 inference
    results = model(frame)
    annotated_frame = results[0].plot()

    # Display the processed frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Cleanly listen for the exit key while refreshing the window layout
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Window closed cleanly.")