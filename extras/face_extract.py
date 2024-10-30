import cv2
import os

def extract_faces_from_video(video_path, output_dir, frame_interval=30):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every nth frame
        if frame_count % frame_interval == 0:
            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            # Save each detected face
            for i, (x, y, w, h) in enumerate(faces):
                face = frame[y:y+h, x:x+w]
                face_filename = os.path.join(output_dir, f'face_{frame_count}_{i}.jpg')
                cv2.imwrite(face_filename, face)
                print(f'Saved: {face_filename}')
        
        frame_count += 1

    # Release the video capture object
    cap.release()
    print('Face extraction complete.')

# Example usage
video_path = 'people.mp4'  # Replace with your video file path
output_dir = 'extracted_faces'  # Directory to save extracted faces
extract_faces_from_video(video_path, output_dir, frame_interval=30)  # Adjust frame_interval as needed
