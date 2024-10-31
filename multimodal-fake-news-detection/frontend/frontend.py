import streamlit as st
import torch
from transformers import DebertaV2Tokenizer
from torchvision import transforms
from PIL import Image
import os
import cv2
import moviepy.editor as mp
import speech_recognition as sr

# Load your models 
text_encoder = torch.load("text_encoder.pth").to("cuda" if torch.cuda.is_available() else "cpu")
image_encoder = torch.load("image_encoder.pth").to("cuda" if torch.cuda.is_available() else "cpu")
fusion_model = torch.load("fusion_model.pth").to("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to preprocess and extract text from video
def extract_text_from_video(video_path):
    audio_dir = "audio_chunks"
    os.makedirs(audio_dir, exist_ok=True)
    
    clip = mp.VideoFileClip(video_path)
    audio_path = os.path.join(audio_dir, "temp_audio.wav")
    clip.audio.write_audiofile(audio_path)

    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)

    try:
        transcription = recognizer.recognize_google(audio_data)
        return transcription
    except sr.UnknownValueError:
        return "[Speech not recognized]"
    except sr.RequestError:
        return "[API unavailable]"

# Function to preprocess and extract images from video
def extract_images_from_video(video_path, output_dir="extracted_images", frame_interval=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    image_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            img_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(img_filename, frame)
            image_list.append(img_filename)

        frame_count += 1

    cap.release()
    return image_list

st.title("Misinformation Detection Tool")

video_file = st.file_uploader("Upload a video file", type=["mp4"])

if video_file is not None:
    # Save uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    # Extract text from video
    st.write("Extracting text from video...")
    transcribed_text = extract_text_from_video("temp_video.mp4")
    st.write("Transcribed Text:")
    st.write(transcribed_text)

    # Extract images from video
    st.write("Extracting images from video...")
    extracted_images = extract_images_from_video("temp_video.mp4")

    # Process extracted images through the image encoder
    image_features = []
    for img_path in extracted_images:
        image = Image.open(img_path).convert("RGB")
        image_input = image_transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        features = image_encoder(image_input)
        image_features.append(features)

    image_features = torch.stack(image_features)
    encoding = tokenizer(transcribed_text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"].squeeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    attention_mask = encoding["attention_mask"].squeeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    text_features = text_encoder(input_ids, attention_mask)

    # Combine text and image features
    combined_features = torch.cat((text_features.unsqueeze(0), image_features.mean(dim=0).unsqueeze(0)), dim=1)

    # Classify as misinformation or not
    with torch.no_grad():
        output = fusion_model(combined_features)
        prediction = torch.argmax(output, dim=1).item()

    # Display the classification result
    if prediction == 0:
        st.write("Certainly Fake")
    elif prediction == 1:
        st.write("Probably Fake")
    elif prediction == 2:
        st.write("Probably Real")
    else:
        st.write("Classification Result: Real")
