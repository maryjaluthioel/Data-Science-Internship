from ultralytics import YOLO
import cv2
import time

print("[SYSTEM] Loading YOLOv8 architecture weights...")
# Using the lightweight nano model for high frame rates
model = YOLO("yolov8n.pt")

# --- AUTO HARDWARE WEBCAM INDEX SCANNER ---
print("[SYSTEM] Scanning system hardware for an active video webcam...")
cap = None

for camera_index in [0, 1, 2]:
    print(f"Testing Camera Index: {camera_index}...")
    temp_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    time.sleep(1.0) # Give lens sensor time to wake up cleanly
    
    ret, test_frame = temp_cap.read()
    if temp_cap.isOpened() and ret and test_frame is not None:
        print(f"[SUCCESS] Found functional webcam stream at Index ID: {camera_index}")
        cap = temp_cap
        break
    temp_cap.release()

if cap is None:
    print("[ERROR] Could not find any working webcam stream.")
    exit()
# ------------------------------------------

print("[SYSTEM] Single-Object Tracking Window active. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        cv2.waitKey(1)
        continue

    # Run object tracking inference
    results = model(frame)
    
    # Isolate prediction bounding data configurations
    boxes = results[0].boxes
    
    # Check if any objects were discovered in the frame
    if len(boxes) > 0:
        # Sort objects by highest confidence score so the absolute best prediction is index 0
        best_box_index = boxes.conf.argmax().item()
        top_prediction = boxes[best_box_index]
        
        # Extract location coordinates for the single best target object
        xyxy = top_prediction.xyxy[0].cpu().numpy().astype(int)
        conf = top_prediction.conf[0].item()
        cls_id = int(top_prediction.cls[0].item())
        label_name = model.names[cls_id]
        
        # Draw a single box rectangle frame over the highest confidence object
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 3)
        
        # Render tag label text overhead
        caption_text = f"{label_name} {conf:.2f}"
        cv2.putText(frame, caption_text, (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Display feedback when no targets are within view frame
        cv2.putText(frame, "Searching for objects...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Render frame output pipeline to layout
    cv2.imshow("YOLOv8 Single-Object Tracking", frame)

    # Refresh GUI and check for exit tap 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Camera device closed cleanly.")