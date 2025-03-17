import time
import json
import queue
import threading
import vosk
import sounddevice as sd
import cv2
import numpy as np
import os
import statistics
import ffmpeg
from PIL import Image, ImageDraw, ImageFont
import langid  # Language detection library

# Load both English and Telugu models
ENGLISH_MODEL_PATH = "path/to/vosk-model-en-us-0.22-lgraph"
TELUGU_MODEL_PATH = "path/to/vosk-model-te"

if not os.path.exists(ENGLISH_MODEL_PATH) or not os.path.exists(TELUGU_MODEL_PATH):
    print("Please download both English and Telugu models.")
    exit()

english_model = vosk.Model(ENGLISH_MODEL_PATH)
telugu_model = vosk.Model(TELUGU_MODEL_PATH)

def detect_language(text):
    """Detect language using langid."""
    lang, _ = langid.classify(text)
    return lang

def process_final_result(text):
    """Process the final result: remove 'the', capitalize, and add punctuation."""
    words = text.split()
    if not words:
        return text
    if words[0].lower() == "the":
        words = words[1:]
    if words and words[-1].lower() == "the":
        words = words[:-1]

    processed_text = ' '.join(words)
    if processed_text:
        processed_text = processed_text[0].upper() + processed_text[1:]
    if processed_text and processed_text[-1] not in ".!?":
        processed_text += "."

    return processed_text

def recognize_speech_from_video(audio_queue, results, samplerate=16000, framerate=60):
    """Recognize speech from video and automatically detect language."""
    rec_en = vosk.KaldiRecognizer(english_model, samplerate)
    rec_te = vosk.KaldiRecognizer(telugu_model, samplerate)
    total_audio_duration = 0.0

    while True:
        data = audio_queue.get()
        if data is None:
            break
        total_audio_duration += len(data) / (samplerate * 2)
        frame_number = int(total_audio_duration * framerate)

        rec = rec_en if rec_en.AcceptWaveform(data) else rec_te
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result['text']:
                text = result['text']
                lang = detect_language(text)
                processed_text = process_final_result(text)
                print(f"Frame {frame_number}: [{lang.upper()}] {processed_text}")
                results.append((frame_number, lang, processed_text))

def render_text_pil(frame, text, position, font_path="path/to/telugu_font.ttf"):
    """Render text using PIL and convert it back to OpenCV format."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_size = 40
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Telugu font not found. Defaulting to system font.")
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def main():
    video_path = "path/to/input_video.mp4"

    if not os.path.exists(video_path):
        print("Video file not found.")
        return

    audio_queue = queue.Queue()
    results = []

    def extract_audio():
        process = (
            ffmpeg.input(video_path)
            .output('pipe:', format='wav', ac=1, ar='16k')
            .run_async(pipe_stdout=True)
        )
        while True:
            in_bytes = process.stdout.read(4000)
            if not in_bytes:
                break
            audio_queue.put(in_bytes)
        audio_queue.put(None)
        process.wait()

    threading.Thread(target=extract_audio, daemon=True).start()
    recognize_speech_from_video(audio_queue, results)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_with_subtitles.mp4", fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        for res_frame, lang, text in results:
            if abs(frame_idx - res_frame) < 5:
                position = (50, height - 100)
                frame = render_text_pil(frame, f"[{lang.upper()}] {text}", position)

        out.write(frame)

    cap.release()
    out.release()
    print("Video processing complete.")

if __name__ == "__main__":
    main()
