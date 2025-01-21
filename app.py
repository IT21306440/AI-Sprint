from flask import Flask, render_template, Response, jsonify
import random
import time
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov8n-custom.pt')  # Fine-tuned YOLO model

# Open video captures
  # Safety monitoring
camera_storage = cv2.VideoCapture('warehouse_video.mp4')  # Storage monitoring
camera_warehouse = cv2.VideoCapture('Duffys order..Smooth pace High percent!.mp4')
if not camera_warehouse.isOpened():
    print("Error: Could not open warehouse_video.mp4")
if not camera_storage.isOpened():
    print("Error: Could not open storage_video.mp4")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Sample safety alerts data
current_alerts = [
    {"message": "No Helmet Detected in Zone A", "severity": "High"},
    {"message": "Blocked Pathway Detected in Zone B", "severity": "Medium"},
]

# Function to detect drunk behavior
def detect_drunk_behavior(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extract keypoints for analysis
        keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        # Analyze movement patterns (e.g., swaying)
        sway_threshold = 0.05
        swaying = np.std([kp[0] for kp in keypoints]) > sway_threshold
        return swaying
    return False

# Function to generate safety monitoring frames
# Function to generate safety monitoring frames
def generate_frames(camera, area_name):
    while True:
        success, frame = camera.read()
        if not success:
            print(f"Error: Could not read frame from {area_name}. Stopping stream.")
            break

        try:
            # Run YOLO object detection
            results = model.predict(frame, conf=0.5)  # Adjust confidence threshold if needed
            annotated_frame = frame.copy()

            # Annotate the frame with bounding boxes and labels
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    confidence = box.conf[0]  # Confidence score
                    label = box.cls  # Class index
                    label_name = model.names[int(label)]  # Class name

                    # Draw bounding box and label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{label_name} ({confidence:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame to JPEG format
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Yield the frame for video streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error processing frame for {area_name}: {e}")
            continue

    camera.release()
    print(f"{area_name} video stream stopped and camera released.")


# Function to generate storage monitoring frames
def generate_storage_frames():
    while True:
        success, frame = camera_storage.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to simulate IoT readings for temperature, air quality, vibration, and alcohol level
def simulate_iot_readings():
    temperature = random.uniform(18, 35)
    air_quality = random.choice(["Good", "Moderate", "Poor"])
    vibration_level = random.uniform(0, 10)
    alcohol_level = random.uniform(0, 0.08)  # Simulated BAC

    recommendations = []
    if temperature > 25:
        recommendations.append("Lower the storage room temperature below 25°C.")
    if air_quality == "Poor":
        recommendations.append("Check air purifiers or ventilation systems.")
    if vibration_level > 7:
        recommendations.append("Inspect machinery to reduce vibrations.")
    if alcohol_level > 0.05:
        recommendations.append("Employee intoxicated: Take immediate action.")

    return {
        "temperature": round(temperature, 2),
        "air_quality": air_quality,
        "vibration_level": round(vibration_level, 2),
        "alcohol_level": round(alcohol_level, 3),
        "recommendations": recommendations,
    }

# Function to generate a bar chart for safety violations
def generate_violation_chart():
    labels = ['No Helmet', 'Blocked Path', 'Unsafe Lifting', 'Drunk Behavior']
    values = [10, 5, 7, 2]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['red', 'blue', 'orange', 'purple'])
    plt.xlabel('Violation Type')
    plt.ylabel('Number of Incidents')
    plt.title('Safety Violations Summary')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return chart_data

# Function to generate a heatmap for warehouse risk
def generate_heatmap():
    data = np.random.rand(10, 10)

    plt.figure(figsize=(6, 4))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Risk Level')
    plt.title('Warehouse Risk Heatmap')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    heatmap_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return heatmap_data

# Function to generate a prediction chart for future incidents
def generate_prediction_chart():
    dates = [f"Day {i}" for i in range(1, 11)]
    incidents = [random.randint(0, 10) for _ in range(10)]

    plt.figure(figsize=(6, 4))
    plt.plot(dates, incidents, marker='o', color='purple')
    plt.xlabel('Days')
    plt.ylabel('Incidents')
    plt.title('Safety Incident Prediction')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return chart_data

@app.route('/')
def index():
    return render_template('index.html', alerts=current_alerts)

@app.route('/video_feed_warehouse')
def video_feed_warehouse():
    return Response(generate_frames(camera_warehouse, "Warehouse Area"), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_storage')
def video_feed_storage():
    return Response(generate_frames(camera_storage, "Storage Area"), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/iot_readings')
def iot_readings():
    readings = simulate_iot_readings()
    return jsonify(readings)

@app.route('/analytics')
def analytics():
    violation_chart = generate_violation_chart()
    heatmap_chart = generate_heatmap()
    prediction_chart = generate_prediction_chart()
    return render_template('analytics.html', 
                           violation_chart=violation_chart, 
                           heatmap_chart=heatmap_chart, 
                           prediction_chart=prediction_chart)

if __name__ == "__main__":
    app.run(debug=True)
