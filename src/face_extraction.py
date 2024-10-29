import cv2
import os
from config.settings import FACE_OUTPUT_PATH, FRAME_INTERVAL, IMAGE_SIZE

def extract_faces_from_video(video_path):
    """
    Extracts faces from video frames at specified intervals using OpenCV.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: Paths to saved face images.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in detected_faces:
                face = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face, IMAGE_SIZE)
                face_path = os.path.join(FACE_OUTPUT_PATH, f'face_{frame_count}.jpg')
                cv2.imwrite(face_path, face_resized)
                faces.append(face_path)
        frame_count += 1

    cap.release()
    return faces
