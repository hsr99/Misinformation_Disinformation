import os
import speech_recognition as sr
import moviepy.editor as mp
from config.settings import AUDIO_CHUNK_PATH

def extract_text_from_video(video_path):
    """
    Extracts text from audio in a video using Google Speech Recognition.

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: A dictionary with timestamps as keys and transcriptions as values.
    """
    recognizer = sr.Recognizer()
    video = mp.VideoFileClip(video_path)
    text_transcriptions = {}

    for i in range(0, int(video.duration), 60):
        audio_chunk = f"{AUDIO_CHUNK_PATH}/audio_{i}.wav"
        video.subclip(i, min(i + 60, video.duration)).audio.write_audiofile(audio_chunk)
        
        with sr.AudioFile(audio_chunk) as source:
            audio_data = recognizer.record(source)
        try:
            text_transcriptions[i] = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text_transcriptions[i] = "[Speech not recognized]"
    
    return text_transcriptions
