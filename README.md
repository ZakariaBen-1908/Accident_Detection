# Accident_Detection
# RoadGuard: Real-Time Accident System

RoadGuard is an intelligent real-time surveillance system that uses computer vision and deep learning to detect vehicle accidents in video footage. It can display real-time alerts and send email notifications with image evidence of detected accidents.

## 🧠 Core Features

- Real-time accident detection using a YOLOv8 model.
- Visual alerts displayed on the video feed.
- Optional email notifications with image snapshots of accidents.
- Uses `cvzone`, `OpenCV`, and `ultralytics` YOLOv8.
- Easily extendable to include speed estimation and vehicle tracking.

---

## 📂 Project Structure

Car_Accident/
├── accident.pt # Pretrained YOLO model for accident detection
├── coco1.txt # Classes used by the model (1 per line)
├── video.mp4 # Input video for detection
├── main.py # Real-time detection with display alerts
├── email_alert.py # Detection + Email notification with image
├── tracker.py # (Optional) Tracking support


## 🔧 Requirements

Make sure you have Python 3.7+ installed, and then install the required packages:

pip install opencv-python cvzone ultralytics numpy

If using email notifications (email_alert.py), you'll also need:

pip install secure-smtplib

Ensure you have access to a Gmail account. If you have 2FA enabled, use an App Password.

🚀 How to Use

1. main.py — Run with Real-Time Detection Only
2. python main.py

Displays video with bounding boxes and accident alerts.

3. email_alert.py — Run with Email Notification
4. Update the following fields in email_alert.py:

sender_email = "your_gmail@gmail.com"

receiver_email = "target_email@gmail.com"

password = "your_app_password"

Then run:

python email_alert.py

Alerts with visual boxes and sends email with a screenshot when an accident is detected.

🧪 Demo Video (Optional)

If you want to provide a demo, include a link to a hosted video demonstrating the system in action.

The detection class in coco1.txt should include a line with "Accident Detection !" (or the same name used in model output).

You can stop the program anytime by pressing q.

📬 Future Improvements

Integrate GPS module for location tracking.

Real-time object speed estimation.

Expand dataset for better accuracy.

Add cloud logging of events.

👨‍💻 Author

Zakaria BENCHEIKH

Feel free to connect or contribute!
