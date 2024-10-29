import cv2
import pytesseract
from fuzzywuzzy import fuzz

def extract_text_from_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray_image)
    cleaned_text = extracted_text.strip()
    return cleaned_text

def filter_duplicates(texts):
    unique_texts = []

    for current_text in texts:
        # Flag to track if we find a match
        is_duplicate = False
        
        for unique_text in unique_texts:
            similarity_ratio = fuzz.partial_ratio(current_text, unique_text)

            # Check for similar texts based on the threshold
            if similarity_ratio > 85:  # Adjusted threshold
                is_duplicate = True
                
                # Keep the longer text if it's more informative
                if len(current_text) > len(unique_text):
                    unique_texts.remove(unique_text)
                    unique_texts.append(current_text)
                break  # No need to check other unique texts

        # If no duplicates found, add to unique list
        if not is_duplicate:
            unique_texts.append(current_text)

    return unique_texts



def video_extract(video_path):
    cap = cv2.VideoCapture(video_path)
    keyframe_texts = []
    frame_number = 0
    keyframe_interval = 30  # Adjust as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract text from the frame
        extracted_text = extract_text_from_image(frame)
        if extracted_text:  # Check if text was extracted
            keyframe_texts.append(extracted_text)

        frame_number += keyframe_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    cap.release()
    
    # Filter out duplicates and return the cleaned unique texts
    unique_texts = filter_duplicates(keyframe_texts)
    return '\n\n'.join(unique_texts)

# Example usage
print(video_extract('Test_video.mp4'))
