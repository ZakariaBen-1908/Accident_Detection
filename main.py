import cv2
import cvzone
from ultralytics import YOLO
import os

path = r"Car_Accident"
os.chdir(path)

# Model
model = YOLO(r"accident.pt")

# Data
df = open(r"coco1.txt", "r")
classes = df.read().split("\n")

cap = cv2.VideoCapture(r"video.mp4")

# Store previous centers, timestamps, and speeds
prev_centers = {}
prev_times = {}

fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video

# Flag to track if an accident alert has been printed
accident_alert_printed = False
accident_detected_frames = 0
cooldown_limit = 30  # Number of frames to wait before resetting alert flag

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not load frame")
        break

    frame = cv2.resize(frame, (1020, 700))

    # Get the timestamp for the current frame (in seconds)
    results = model.predict(frame)

    accident_detected = False

    for result in results:
        for re in result.boxes:
            x1, y1, x2, y2 = map(int, re.xyxy[0])
            if int(re.cls[0]) < len(classes):
                name = classes[int(re.cls[0])]
                ww = x2 - x1
                hh = y2 - y1

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, name, (x1 + 10, y1 - 10), scale=1, thickness=1,
                                   colorR=(0, 255, 0), colorT=(0, 0, 0))

                # Check if an accident is detected
                if "Accident Detection !" in name:
                    accident_detected = True
                    # Highlight accident detection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f"Accident Detected!", (x1 + 10, y1 - 10), scale=1,
                                       thickness=1, colorR=(0, 0, 255), colorT=(0, 0, 0))
                    
                    # Print alert if not already printed
                    if not accident_alert_printed:
                        print("⚠️ Accident Detected! Check the video feed.")
                        accident_alert_printed = True

    # Cooldown to reset alert flag
    if not accident_detected:
        accident_detected_frames += 1
        if accident_detected_frames > cooldown_limit:
            accident_alert_printed = False
            accident_detected_frames = 0
    else:
        accident_detected_frames = 0  # Reset frame counter if accident is detected

    cv2.imshow("RoadGuard Real-Time Accident and Speed Detection System", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
