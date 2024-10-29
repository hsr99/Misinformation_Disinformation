import cv2
import pytesseract

video_path = "Test_vide0.mp4"

def video_extract(video_path):
    cap = cv2.VideoCapture(video_path)

    previous_frame = None
    keyframe_texts = []
    frame_number = 0
    keyframe_interval = 30  # Adjust as needed to capture key changes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for easier comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if the frame is significantly different from the previous keyframe
        if previous_frame is None or cv2.norm(previous_frame, gray_frame, cv2.NORM_L2) > 1000:  # Adjust threshold as needed
            # Extract text from the keyframe
            text = pytesseract.image_to_string(frame)
            keyframe_texts.append((frame_number, text))
            
            # Update previous frame to the current frame
            previous_frame = gray_frame

        frame_number += keyframe_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Skip to the next keyframe interval

    # Release the video capture object
    cap.release()

    # Display or store the extracted text from keyframes
    for frame_num, text in keyframe_texts:
        print(f"Keyframe {frame_num}: {text}\n")
        
     
