from ultralytics import YOLO
import cv2
import time

print("[SYSTEM] Loading YOLOv8 architecture weights...")
model = YOLO("yolov8n.pt")

# --- AUTO HARDWARE WEBCAM INDEX SCANNER ---
print("[SYSTEM] Scanning system hardware for an active video webcam...")
cap = None

# Loop through common camera indexes (0 = default, 1 = external/virtual, 2 = integrated secondary)
for camera_index in [0, 1, 2]:
    print(f"Testing Camera Index: {camera_index}...")
    temp_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # Give the hardware lens a brief moment to open up and stabilize
    time.sleep(1.0) 
    
    ret, test_frame = temp_cap.read()
    if temp_cap.isOpened() and ret and test_frame is not None:
        print(f"[SUCCESS] Found functional webcam stream at Index ID: {camera_index}")
        cap = temp_cap
        break
    temp_cap.release()

if cap is None:
    print("[ERROR] Could not find any working webcam stream. Make sure no other app (like Zoom/Teams) is using your camera.")
    exit()
# ------------------------------------------

print("[SYSTEM] Window launching now. Click the popup and press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        cv2.waitKey(1)
        continue

    # Run object tracking inference
    results = model(frame)
    annotated_frame = results[0].plot()

    # Render frame to graphic window
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Refresh GUI layout and listen for exit command 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Camera device released safely.")