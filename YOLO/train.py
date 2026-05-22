import cv2
import mediapipe as mp

# Access legacy media processing solutions
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create face detection object
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Verify VS Code can access the webcam resource
if not cap.isOpened():
    print("Error: Could not open webcam. Check background tasks.")
    exit()

print("System Ready! Press 'q' on your keyboard to close the display.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Empty frame received. Exiting layout...")
        break

    # Convert to RGB (MediaPipe requirement)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to isolate face coordinates
    results = face_detection.process(rgb)

    # Draw bounding boxes and target points
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Output feed stream
    cv2.imshow("Mediapipe Face Detection", frame)

    # Exit validation 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()