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


def process_final_result(text):
    """Process the final result: remove 'the', capitalize the first letter, and add a period at the end."""
    words = text.split()
    if len(words) == 0:
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

def recognize_speech_from_video(model, audio_queue, results, samplerate=16000, framerate=60):
    """Recognize speech from the video in real-time."""
    rec = vosk.KaldiRecognizer(model, samplerate)
    total_audio_duration = 0.0

    while True:
        data = audio_queue.get()
        if data is None:
            break
        total_audio_duration += len(data) / (samplerate * 2)  
        frame_number = int(total_audio_duration * framerate)

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result['text']:
                transcription = process_final_result(result['text']) 
                print(f"Frame {frame_number}: {transcription}") 
                results.append((frame_number, 'full', transcription))
        elif rec.PartialResult():
            partial_result = json.loads(rec.PartialResult())
            if partial_result['partial']:
                transcription = partial_result['partial']
                print(f"Frame {frame_number}: Partial result: {transcription}")  
                results.append((frame_number, 'partial', transcription))

def display_transcriptions(frame, transcriptions, partial_transcription, x0, y0, dy, prev_length_difference, max_length=40):
    """Combine finalized and partial transcriptions and display them on the frame."""
    
    lines = []
    
    combined_transcription = ' '.join(transcriptions) + process_final_result(' '.join(partial_transcription))
    words = combined_transcription.split()
    
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_length:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)

    original_length = len(lines)
    lines = lines[-3:]
    length_difference = original_length - len(lines)

    if length_difference - prev_length_difference == 1:
        print("changed")
        y0 += dy * 2.5

    textScaler = 3.0
    border = 2
    thickness = 5
    
    for i, line in enumerate(lines):
        y = y0 + i * dy * textScaler
        (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * textScaler, 1)
        adjusted_x0 = int(x0 - text_width // 2)
        cv2.putText(frame, line, (adjusted_x0-border, int(y-border)), cv2.FONT_HERSHEY_SIMPLEX, .5 * textScaler, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, line, (adjusted_x0+border, int(y+border)), cv2.FONT_HERSHEY_SIMPLEX, .5 * textScaler, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, line, (adjusted_x0+border, int(y-border)), cv2.FONT_HERSHEY_SIMPLEX, .5 * textScaler, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, line, (adjusted_x0-border, int(y+border)), cv2.FONT_HERSHEY_SIMPLEX, .5 * textScaler, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, line, (adjusted_x0-border, int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5 * textScaler, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, line, (adjusted_x0+border, int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5 * textScaler, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, line, (adjusted_x0, int(y-border)), cv2.FONT_HERSHEY_SIMPLEX, .5 * textScaler, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, line, (adjusted_x0, int(y+border)), cv2.FONT_HERSHEY_SIMPLEX, .5 * textScaler, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, line, (adjusted_x0, int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5 * textScaler, (0, 0, 255), thickness-2, cv2.LINE_AA)
        
    return lines, length_difference

def compute_median(positions):
    """Compute the median of the last five positions."""
    if len(positions) >= 15:
        median_value = statistics.median(positions[-15:])
    else:
        median_value = statistics.median(positions)
    
    return median_value

def compute_smoothed_position(medians):
    """Compute the average of the last four median values."""
    if len(medians) >= 10:
        return sum(medians[-10:]) / 10
    return sum(medians) / len(medians)

def main():

    # model_path = "C:/Storage/vosk-model-en-us-0.22-lgraph" #ENGL model
    model_path = "D:/study/mini project/transcriptions/vosk-model-en-us-0.22-lgraph"

    if not os.path.exists(model_path):
        print(f"Please download the model from https://alphacephei.com/vosk/models and unpack it as '{model_path}'")
        return
    model = vosk.Model(model_path)

    audio_queue = queue.Queue()

    results = []

    # video_path = "D:/study/mini project/transcriptions/input_video2.mp4" 
    video_path = "D:/study/mini project/transcriptions/input_video2.mp4"

    def extract_audio():
        process = (
            ffmpeg
            .input(video_path)
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
    recognize_speech_from_video(model, audio_queue, results)

    print("Audio processing complete. Starting video processing.")

    cap = cv2.VideoCapture(video_path)

    transcriptions = ["", "", ""]
    partial_transcription = [""]
    lock = threading.Lock()
    running = True

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_centers_x = []
    face_centers_y = []
    face_sizes_y = []
    median_x_values = []
    median_y_values = []
    timer = 0

    prev_length_difference = 0

    frame_idx = 0


    test_scale_factor = 1  

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*test_scale_factor 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*test_scale_factor 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('D:/study/mini project/transcriptions/output_with_subtitles.mp4', fourcc, fps, (width, height))

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            if timer > 0:
                timer -= 2

            frame = cv2.resize(frame, None, fx=test_scale_factor, fy=test_scale_factor)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            scale_factor = 0.25  
            small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
            
            faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                if w > 100:
                    cv2.rectangle(frame, (x*4, y*4), (x*4+w*4, y*4+h*4), (255, 255, 255), 4)
                    center_x = x * 4 + 4 * w // 2
                    center_y = y * 4 + 4 * h // 2
                    face_centers_x.append(center_x)
                    face_centers_y.append(center_y)
                    face_sizes_y.append(4 * h // 2)
                    if len(face_centers_x) > 5:
                        face_centers_x.pop(0)
                    if len(face_centers_y) > 5:
                        face_centers_y.pop(0)
                    
            x0 = 10
            y0 = 10
            chinY = 130

            if face_centers_x and face_centers_y:
                median_x = compute_median(face_centers_x)
                median_y = compute_median(face_centers_y)
                chinY = compute_median(face_sizes_y)
                median_x_values.append(median_x)
                median_y_values.append(median_y)

                x0 = compute_smoothed_position(median_x_values)
                y0 = compute_smoothed_position(median_y_values)

                #cv2.circle(frame, (int(x0), int(y0)), 2, (0, 255, 0), -1)

            while results and results[0][0] <= frame_idx:
                _, result_type, transcription = results.pop(0)
                with lock:
                    if result_type == 'full':
                        transcriptions.append(transcription)
                        if len(transcriptions) > 3:
                            transcriptions.pop(0)
                        partial_transcription.clear()  
                    elif result_type == 'partial':
                        partial_transcription.clear()
                        partial_transcription.append(transcription)

            dy = 20
            with lock:
                lines, length_difference = display_transcriptions(frame, transcriptions, partial_transcription, x0, y0 + chinY + 30 + timer, dy, prev_length_difference)

            if length_difference - prev_length_difference == 1:
                print("changed2")
                timer += dy * 2.5
            
            prev_length_difference = length_difference

            out.write(frame)
            
            frame2 = frame.copy()
            frame2 = cv2.resize(frame2, None, fx=.25, fy=.25)


            cv2.imshow('frame', frame2)

            if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
                running = False
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

            frame_idx += 1
            
            height, width = frame.shape[:2]

    except KeyboardInterrupt:
        print("\nDone.")
        
    output_video_path = 'D:/study/mini project/transcriptions/output_with_subtitles.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    running = False
    

    input_video = ffmpeg.input(output_video_path)
    input_audio = ffmpeg.input(video_path).audio
    ffmpeg.output(input_video, input_audio, 'D:/study/mini project/transcriptions/final_output_with_audio.mp4', vcodec='copy', acodec='aac').run()

if __name__ == "__main__":
    main()