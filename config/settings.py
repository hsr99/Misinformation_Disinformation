import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Paths
VIDEO_PATH = 'path_to_your_video.mp4'
AUDIO_CHUNK_PATH = 'data/audio_chunks'
FACE_OUTPUT_PATH = 'data/extracted_faces'
TEXT_MODEL_NAME = "microsoft/deberta-v3-large"

# Other configurations
FRAME_INTERVAL = 30
IMAGE_SIZE = (224, 224)
