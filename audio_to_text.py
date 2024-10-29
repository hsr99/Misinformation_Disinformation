import cv2
import pytesseract

def remove_redundant_lines(text):
    # Split text into lines and remove duplicates while preserving order
    unique_lines = list(dict.fromkeys(text.splitlines()))
    
    # Join the unique lines back into a single string with appropriate spacing
    return '\n'.join(unique_lines)

def video_extract(video_path):
    cap = cv2.VideoCapture(video_path)

    previous_frame = None
    keyframe_texts = []
    frame_number = 0
    keyframe_interval = 40  # Adjust as needed to capture key changes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for easier comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if the frame is significantly different from the previous keyframe
        if previous_frame is None or cv2.norm(previous_frame, gray_frame, cv2.NORM_L2) > 1000:  # Adjust threshold as needed
            # Extract text from the keyframe
            text = pytesseract.image_to_string(frame).strip()
            if text:  # Only add non-empty text
                keyframe_texts.append(text)
            
            # Update previous frame to the current frame
            previous_frame = gray_frame

        frame_number += keyframe_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Skip to the next keyframe interval

    # Release the video capture object
    cap.release()
    
    # Join all extracted texts and remove duplicates
    combined_text = '\n'.join(keyframe_texts)  # Combine texts with double newline for readability
    cleaned_text = remove_redundant_lines(combined_text)  # Clean the text to remove duplicates

    return cleaned_text

print(video_extract('Test_video.mp4'))
