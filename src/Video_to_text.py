import speech_recognition as sr
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

# Set your video file path
video_path = ""

# Output directories for video chunks and audio files
chunk_dir = "chunks"
audio_dir = "converted"
os.makedirs(chunk_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)
clip = mp.VideoFileClip(video_path)
num_seconds_video = int(clip.duration)
print(f"The video is {num_seconds_video} seconds long")
intervals = list(range(0, num_seconds_video + 1, 60))
transcriptions = {}

for i in range(len(intervals) - 1):
    start_time = intervals[i] - 2 * (intervals[i] != 0)  # Small overlap if not first chunk
    end_time = intervals[i + 1]
    chunk_filename = f"{chunk_dir}/cut_{i + 1}.mp4"
    ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=chunk_filename)
    
    chunk_clip = mp.VideoFileClip(chunk_filename)
    audio_filename = f"{audio_dir}/audio_{i + 1}.wav"
    chunk_clip.audio.write_audiofile(audio_filename)
    
    # Transcribe the audio using SpeechRecognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_filename) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)
    
    try:
        # Use Google Speech Recognition to transcribe audio
        transcription = recognizer.recognize_google(audio_data)
        transcriptions[f'chunk_{i + 1}'] = transcription
    except sr.UnknownValueError:
        transcriptions[f'chunk_{i + 1}'] = "[Speech not recognized]"
    except sr.RequestError:
        transcriptions[f'chunk_{i + 1}'] = "[API unavailable]"

for chunk, text in transcriptions.items():
    print(f"{chunk}: {text}")





