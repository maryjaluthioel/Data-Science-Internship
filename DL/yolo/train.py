import cv2
import mediapipe as mp
import time

# Access legacy media processing solutions
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create face detection object
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --- AUTO HARDWARE WEBCAM INDEX SCANNER ---
print("[SYSTEM] Scanning system hardware for an active video webcam...")
cap = None

for camera_index in [0, 1, 2]:
    print(f"Testing Camera Index: {camera_index}...")
    temp_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # Give hardware lens a brief moment to stabilize
    time.sleep(1.0) 
    
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

print("System Ready! Press 'q' on your keyboard to close the display.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        cv2.waitKey(1)
        continue

    # Convert to RGB (MediaPipe requirement)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to isolate face coordinates
    results = face_detection.process(rgb)

    # Draw bounding boxes and target points if faces are found
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Render frame to graphic window
    cv2.imshow("MediaPipe Face Detection", frame)

    # Refresh GUI layout and listen for exit command 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Camera device released safely.")