# Import necessary libraries
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import DebertaV2Tokenizer, DebertaV2Model
from torchvision import models, transforms
from PIL import Image
import cv2
import json
import streamlit as st

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define Text Encoder (DeBERTaV3 Large)
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-large")
        self.proj = nn.Linear(1024, 512)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.proj(outputs[:, 0, :])  # Use [CLS] token for text representation

# Define Image Encoder (DenseNet-121)
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1024, 512)

    def forward(self, images):
        features = self.features(images)
        pooled = self.avgpool(features).view(features.size(0), -1)
        return self.proj(pooled)

# Define Multimodal Model with Transformer Encoder for Fusion
class MultimodalTransformerModel(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=4):
        super(MultimodalTransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_features, image_features):
        combined_features = torch.stack([text_features, image_features], dim=1)
        fused_features = self.transformer_encoder(combined_features).mean(dim=1)
        return self.classifier(fused_features)

# Load and initialize tokenizer and image transformations
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Text and image preprocessing functions
def preprocess_text(text):
    encoding = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    return encoding["input_ids"].squeeze(0).to(device), encoding["attention_mask"].squeeze(0).to(device)

def preprocess_image(image):
    return image_transform(image).unsqueeze(0).to(device)

# Load models and weights
text_encoder = TextEncoder().to(device)
image_encoder = ImageEncoder().to(device)
model = MultimodalTransformerModel().to(device)

# Assuming models are saved in the current directory with these names
text_encoder.load_state_dict(torch.load("best_accuracy_text_encoder.pth"))
image_encoder.load_state_dict(torch.load("best_accuracy_image_encoder.pth"))
model.load_state_dict(torch.load("best_accuracy_multimodal_transformer.pth"))

# Inference function for text and image
def multimodal_predict(text, image):
    input_ids, attention_mask = preprocess_text(text)
    text_features = text_encoder(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
    image_tensor = preprocess_image(image)
    image_features = image_encoder(image_tensor)
    output = model(text_features, image_features)
    return torch.argmax(output, dim=1).item()

# Video processing and deepfake detection
def extract_text_from_video(video_path):
    # Example function for extracting text (can replace with real OCR-based extraction)
    return ["Sample transcription from video."]

def extract_faces_from_video(video_path):
    faces = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Placeholder for face extraction logic
        face_img = Image.fromarray(frame)
        faces.append(face_img)
    cap.release()
    return faces[:5]  # Return top 5 faces as example

def run_video_pipeline(video_path):
    transcriptions = extract_text_from_video(video_path)
    face_images = extract_faces_from_video(video_path)
    frames = [face_images[0]] if face_images else []  # Use extracted face as frame
    
    if frames:
        text_features, image_features = [], []
        for frame in frames:
            input_ids, attention_mask = preprocess_text(transcriptions[0])
            text_feat = text_encoder(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
            text_features.append(text_feat)

            image_tensor = preprocess_image(frame)
            image_feat = image_encoder(image_tensor)
            image_features.append(image_feat)

        text_features = torch.cat(text_features, dim=0)
        image_features = torch.cat(image_features, dim=0)
        prediction = model(text_features, image_features).mean(dim=0)
        return torch.argmax(prediction, dim=0).item(), transcriptions, face_images
    return None, [], []

# Streamlit interface
st.title("Multimodal Fake Detection App")

option = st.sidebar.selectbox(
    "Choose Analysis Mode", 
    ("Text + Image", "Video")
)

if option == "Text + Image":
    st.header("Text + Image Mode")
    
    sample_text = "Dopo aver hackerato la tv di Stato sostituendo immagini di propaganda con quelle vere si sono rivolti direttamente a Putin “Contro di noi non puoi vincere” e ai soldati sul fronte con un appello a deporre le armi. Oggi si può combattere una guerra senza sparare un colpo. Rispetto https://t.co/NcPXUYgzHX"
    text_input = st.text_area("Enter text for analysis", sample_text)
    
    sample_image_path = "Tweet_Images\\1498022438398877704.jpg"  # Replace with an actual sample image path
    sample_image = Image.open(sample_image_path) if os.path.exists(sample_image_path) else None
    image_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    image = Image.open(image_input) if image_input else sample_image
    if image:
        st.image(image, caption="Image for analysis", use_column_width=True)
    
    if st.button("Analyze Text + Image"):
        if text_input and image:
            prediction = multimodal_predict(text_input, image)
            labels = ["Certainly Fake", "Partially Fake", "Partially True", "Certainly True"]
            st.write(f"Prediction: {labels[prediction]}")
        else:
            st.write("Please provide both text and an image.")

elif option == "Video":
    st.header("Video Mode")
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if video_file:
        video_path = os.path.join("uploads", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        if st.button("Analyze Video"):
            prediction, transcriptions, face_images = run_video_pipeline(video_path)
            labels = ["Certainly Fake", "Partially Fake", "Partially True", "Certainly True"]
            st.write(f"Deepfake Prediction: {labels[prediction]}")
            st.write("Extracted Transcriptions:")
            for i, transcription in enumerate(transcriptions):
                st.write(f"{i+1}: {transcription}")
            
            st.write("Extracted Face Images:")
            for i, face in enumerate(face_images):
                st.image(face, caption=f"Face {i+1}", use_column_width=True)
