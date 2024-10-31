import json
from src.text_extraction import extract_text_from_video
from src.face_extraction import extract_faces_from_video
from src.deepfake_model import create_bnn_model
from src.utils import preprocess_video_frames
from config.settings import VIDEO_PATH

def run_pipeline(video_path):
    print("Extracting text from video...")
    transcriptions = extract_text_from_video(video_path)
    
    print("Extracting faces from video...")
    face_images = extract_faces_from_video(video_path)
    
    print("Running deepfake detection...")
    frames = preprocess_video_frames(video_path)
    bnn_model = create_bnn_model()
    prediction = bnn_model.predict(np.expand_dims(frames, axis=0))
    
    results = {
        "transcriptions": transcriptions,
        "faces": face_images,
        "deepfake_prediction": prediction.tolist(),
    }

    with open('results/output.json', 'w') as f:
        json.dump(results, f)
    print("Pipeline complete. Results saved to results/output.json")

run_pipeline(VIDEO_PATH)
