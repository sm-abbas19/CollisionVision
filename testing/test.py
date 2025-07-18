import cv2
import requests

API_URL = "http://localhost:8000/process_frame/"

video_path = "CollisionVision/testing/test.webm"  # Replace with your video file path

cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

    # Send to API
    response = requests.post(API_URL, files=files)
    print(f"Frame {frame_count}: {response.status_code}")
    print(response.json())  # Print the JSON result

    frame_count += 1

    # Optional: limit number of frames for testing
    # if frame_count > 10:
    #     break

cap.release()