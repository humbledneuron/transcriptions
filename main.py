import time, json, queue, threading, vosk, sounddevice as sd, cv2, numpy as np, os, statistics, ffmpeg, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
import customtkinter as ctk

class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognition and Subtitle Generator")
        self.root.geometry("1200x720")
        self.root.minsize(1200, 720)
        
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        self.setup_variables()
        self.create_ui()
        
    def setup_variables(self):
        self.video_path = ""
        self.output_path = ""
        self.model_path = ""
        self.is_recording = False
        self.is_processing = False
        self.is_live = False
        self.language = tk.StringVar(value="English")
        self.results = []
        self.audio_queue = queue.Queue()
        self.transcriptions = ["", "", ""]
        self.partial_transcription = [""]
        self.lock = threading.Lock()
        self.face_centers_x = []
        self.face_centers_y = []
        self.face_sizes_y = []
        self.median_x_values = []
        self.median_y_values = []
        self.prev_length_difference = 0
        self.timer = 0
        self.text_scale = tk.DoubleVar(value=3.0)
        self.border = tk.IntVar(value=2)
        self.thickness = tk.IntVar(value=5)
        self.subtitle_color = (0, 0, 255)
        self.outline_color = (255, 255, 255)
    
    def create_ui(self):
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(side="left", fill="y", padx=10, pady=10, expand=False)
        
        # Video panel
        self.video_frame = ctk.CTkFrame(main_frame)
        self.video_frame.pack(side="right", fill="both", padx=10, pady=10, expand=True)
        
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)
        
        # Language selection
        lang_frame = ctk.CTkFrame(control_frame)
        lang_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(lang_frame, text="Language:").pack(side="left", padx=5)
        languages = ["English", "Telugu"]
        language_menu = ctk.CTkOptionMenu(lang_frame, values=languages, variable=self.language, command=self.change_language)
        language_menu.pack(side="right", padx=5, fill="x", expand=True)
        
        # Model selection
        model_frame = ctk.CTkFrame(control_frame)
        model_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(model_frame, text="Model:").pack(side="left", padx=5)
        self.model_path_label = ctk.CTkLabel(model_frame, text="Not selected", wraplength=200)
        self.model_path_label.pack(side="right", fill="x", expand=True)
        
        model_button = ctk.CTkButton(control_frame, text="Select Model", command=self.select_model)
        model_button.pack(fill="x", padx=5, pady=5)
        
        # Input selection
        input_frame = ctk.CTkFrame(control_frame)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(input_frame, text="Input Mode:").pack(side="left", padx=5)
        self.live_toggle = ctk.CTkSwitch(input_frame, text="Live Input", command=self.toggle_live)
        self.live_toggle.pack(side="right", padx=5)
        
        # File input options
        self.file_frame = ctk.CTkFrame(control_frame)
        self.file_frame.pack(fill="x", padx=5, pady=5)
        
        file_button = ctk.CTkButton(self.file_frame, text="Select Video", command=self.select_video)
        file_button.pack(fill="x", padx=5, pady=5)
        
        self.video_path_label = ctk.CTkLabel(self.file_frame, text="No video selected", wraplength=200)
        self.video_path_label.pack(fill="x", padx=5, pady=5)
        
        output_button = ctk.CTkButton(self.file_frame, text="Set Output Location", command=self.set_output)
        output_button.pack(fill="x", padx=5, pady=5)
        
        self.output_path_label = ctk.CTkLabel(self.file_frame, text="Default output location", wraplength=200)
        self.output_path_label.pack(fill="x", padx=5, pady=5)
        
        # Appearance settings
        appearance_frame = ctk.CTkFrame(control_frame)
        appearance_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(appearance_frame, text="Subtitle Appearance").pack(padx=5, pady=5)
        
        scale_frame = ctk.CTkFrame(appearance_frame)
        scale_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(scale_frame, text="Text Size:").pack(side="left", padx=5)
        scale_slider = ctk.CTkSlider(scale_frame, from_=1.0, to=5.0, variable=self.text_scale)
        scale_slider.pack(side="right", fill="x", expand=True, padx=5)
        
        color_frame = ctk.CTkFrame(appearance_frame)
        color_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(color_frame, text="Text Color:").pack(side="left", padx=5)
        color_options = ["Red", "Blue", "Green", "Yellow", "White"]
        color_menu = ctk.CTkOptionMenu(color_frame, values=color_options, command=self.change_text_color)
        color_menu.pack(side="right", fill="x", expand=True, padx=5)
        color_menu.set("Blue")
        
        # Action buttons
        buttons_frame = ctk.CTkFrame(control_frame)
        buttons_frame.pack(fill="x", padx=5, pady=10)
        
        self.start_button = ctk.CTkButton(buttons_frame, text="Start", command=self.start_processing, fg_color="green")
        self.start_button.pack(fill="x", padx=5, pady=5)
        
        self.stop_button = ctk.CTkButton(buttons_frame, text="Stop", command=self.stop_processing, fg_color="red", state="disabled")
        self.stop_button.pack(fill="x", padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ctk.CTkLabel(self.root, textvariable=self.status_var, anchor="w")
        status_bar.pack(side="bottom", fill="x", padx=10, pady=5)
        
    def change_language(self, choice):
        self.language.set(choice)
        self.update_status(f"Language changed to {choice}")
    
    def change_text_color(self, choice):
        color_map = {
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "Green": (0, 255, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255)
        }
        self.subtitle_color = color_map.get(choice, (255, 0, 0))
    
    def select_model(self):
        model_dir = filedialog.askdirectory(title="Select Vosk Model Directory")
        if model_dir:
            self.model_path = model_dir
            self.model_path_label.configure(text=os.path.basename(model_dir))
            self.update_status(f"Model selected: {os.path.basename(model_dir)}")
    
    def toggle_live(self):
        self.is_live = self.live_toggle.get()
        if self.is_live:
            self.file_frame.pack_forget()
        else:
            self.file_frame.pack(fill="x", padx=5, pady=5)
    
    def select_video(self):
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi;*.mkv;*.mov")])
        if video_path:
            self.video_path = video_path
            self.video_path_label.configure(text=os.path.basename(video_path))
            self.update_status(f"Video selected: {os.path.basename(video_path)}")
    
    def set_output(self):
        output_path = filedialog.asksaveasfilename(title="Save Output As", defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if output_path:
            self.output_path = output_path
            self.output_path_label.configure(text=os.path.basename(output_path))
            self.update_status(f"Output will be saved as: {os.path.basename(output_path)}")
    
    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def start_processing(self):
        if self.is_live and not self.model_path:
            messagebox.showerror("Error", "Please select a model first")
            return
        
        if not self.is_live and (not self.video_path or not self.model_path):
            messagebox.showerror("Error", "Please select both a video and a model")
            return
        
        if not self.output_path and not self.is_live:
            default_name = "output_" + time.strftime("%Y%m%d_%H%M%S") + ".mp4"
            self.output_path = os.path.join(os.path.dirname(self.video_path), default_name)
            self.output_path_label.configure(text=default_name)
        
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        if self.is_live:
            threading.Thread(target=self.process_live, daemon=True).start()
        else:
            threading.Thread(target=self.process_video, daemon=True).start()
    
    def stop_processing(self):
        self.is_processing = False
        self.is_recording = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.update_status("Processing stopped")
    
    def process_live(self):
        self.is_processing = True
        self.is_recording = True
        self.results = []
        
        model = vosk.Model(self.model_path)
        device_info = sd.query_devices(None, 'input')
        samplerate = int(device_info['default_samplerate'])
        
        def callback(indata, frames, time, status):
            if self.is_recording:
                self.audio_queue.put(bytes(indata))
        
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16', channels=1, callback=callback):
            self.update_status("Recording started")
            
            # Start recognition thread
            threading.Thread(target=self.recognize_speech, args=(model, samplerate, 30), daemon=True).start()
            
            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                self.stop_processing()
                return
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            frame_idx = 0
            
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                self.process_frame(frame, face_cascade, frame_idx)
                frame_idx += 1
                
                # Display frame
                self.display_frame(frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
            
            cap.release()
            self.is_recording = False
            self.update_status("Live processing stopped")
    
    def process_video(self):
        self.is_processing = True
        self.results = []
        
        model = vosk.Model(self.model_path)
        
        # Start audio extraction and recognition
        threading.Thread(target=self.extract_audio, args=(self.video_path,), daemon=True).start()
        threading.Thread(target=self.recognize_speech, args=(model,), daemon=True).start()
        
        self.update_status("Processing video...")
        
        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            self.stop_processing()
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        frame_idx = 0
        
        while self.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            self.process_frame(frame, face_cascade, frame_idx)
            frame_idx += 1
            
            # Write frame to output
            out.write(frame)
            
            # Display frame
            self.display_frame(frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        cap.release()
        out.release()
        
        if self.is_processing:  # If stopped naturally (not by user)
            # Combine video with original audio
            self.combine_video_audio()
        
        self.is_processing = False
        self.update_status("Video processing completed")
    
    def extract_audio(self, video_path):
        process = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='wav', ac=1, ar='16k')
            .run_async(pipe_stdout=True)
        )
        while self.is_processing:
            in_bytes = process.stdout.read(4000)
            if not in_bytes:
                break
            self.audio_queue.put(in_bytes)
        self.audio_queue.put(None)
        process.wait()
    
    def recognize_speech(self, model, samplerate=16000, framerate=60):
        rec = vosk.KaldiRecognizer(model, samplerate)
        total_audio_duration = 0.0
        
        while self.is_processing:
            try:
                data = self.audio_queue.get(timeout=1)
                if data is None:
                    break
                
                total_audio_duration += len(data) / (samplerate * 2)
                frame_number = int(total_audio_duration * framerate)
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result['text']:
                        transcription = self.process_final_result(result['text'])
                        self.results.append((frame_number, 'full', transcription))
                else:
                    partial_result = json.loads(rec.PartialResult())
                    if partial_result['partial']:
                        transcription = partial_result['partial']
                        self.results.append((frame_number, 'partial', transcription))
            except queue.Empty:
                continue
    
    def process_final_result(self, text):
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
    
    def process_frame(self, frame, face_cascade, frame_idx):
        if self.timer > 0:
            self.timer -= 2
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale_factor = 0.25
        small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
        
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Track face positions
        for (x, y, w, h) in faces:
            if w > 40:  # Minimum face size threshold
                face_scale = int(1/scale_factor)
                x, y, w, h = x*face_scale, y*face_scale, w*face_scale, h*face_scale
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                self.face_centers_x.append(center_x)
                self.face_centers_y.append(center_y)
                self.face_sizes_y.append(h // 2)
                
                if len(self.face_centers_x) > 5:
                    self.face_centers_x.pop(0)
                if len(self.face_centers_y) > 5:
                    self.face_centers_y.pop(0)
                if len(self.face_sizes_y) > 5:
                    self.face_sizes_y.pop(0)
        
        x0, y0, chin_y = 10, 10, 130
        
        # Calculate median positions
        if self.face_centers_x and self.face_centers_y:
            median_x = self.compute_median(self.face_centers_x)
            median_y = self.compute_median(self.face_centers_y)
            chin_y = self.compute_median(self.face_sizes_y)
            
            self.median_x_values.append(median_x)
            self.median_y_values.append(median_y)
            
            x0 = self.compute_smoothed_position(self.median_x_values)
            y0 = self.compute_smoothed_position(self.median_y_values)
        
        # Update transcriptions
        while self.results and self.results[0][0] <= frame_idx:
            _, result_type, transcription = self.results.pop(0)
            with self.lock:
                if result_type == 'full':
                    self.transcriptions.append(transcription)
                    if len(self.transcriptions) > 3:
                        self.transcriptions.pop(0)
                    self.partial_transcription.clear()
                elif result_type == 'partial':
                    self.partial_transcription.clear()
                    self.partial_transcription.append(transcription)
        
        # Display transcriptions
        dy = 20
        with self.lock:
            lines, length_difference = self.display_transcriptions(
                frame, self.transcriptions, self.partial_transcription, 
                x0, y0 + chin_y + 30 + self.timer, dy
            )
        
        # Update animation timer
        if length_difference - self.prev_length_difference == 1:
            self.timer += dy * 2.5
        
        # Update previous length difference
        self.prev_length_difference = length_difference
    
    def display_transcriptions(self, frame, transcriptions, partial_transcription, x0, y0, dy):
        lines = []
        max_length = 25 if self.language.get() == "Telugu" else 40
        
        # Combine transcriptions and partial transcription
        combined_transcription = ' '.join(transcriptions) + self.process_final_result(' '.join(partial_transcription))
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
        
        # Calculate the difference in the number of lines
        original_length = len(lines)
        lines = lines[-3:]
        length_difference = original_length - len(lines)
        
        text_scaler = self.text_scale.get()
        border = self.border.get()
        thickness = self.thickness.get()
        
        height, width = frame.shape[:2]
        
        # Select appropriate font based on language
        font_path = None
        if self.language.get() == "Telugu":
            font_path = "path/to/telugu/font.ttf"  # Replace with actual Telugu font path
        
        for i, line in enumerate(lines):
            y = int(y0 + i * dy * text_scaler)
            
            if self.language.get() == "Telugu" and font_path and os.path.exists(font_path):
                # Use PIL for Telugu text
                self.put_text(frame, line, (int(x0), y), font_path, int(20 * text_scaler), self.subtitle_color, self.outline_color, border)
            else:
                # Use OpenCV for English text
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * text_scaler, 1)
                adjusted_x0 = int(x0 - text_width // 2)
                
                # Draw outline
                for dx, dy in [(-border, -border), (border, border), (border, -border), (-border, border), 
                              (-border, 0), (border, 0), (0, -border), (0, border)]:
                    cv2.putText(frame, line, (adjusted_x0 + dx, y + dy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5 * text_scaler, 
                              self.outline_color, thickness, cv2.LINE_AA)
                
                # Draw main text
                cv2.putText(frame, line, (adjusted_x0, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5 * text_scaler, 
                          self.subtitle_color, thickness-2, cv2.LINE_AA)
        
        return lines, length_difference
    
    def put_text(self, image, text, position, font_path, font_size, color, outline_color, border):
        # Convert to RGB for PIL
        cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        x, y = position
        
        # Draw outline
        for dx, dy in [(-border, -border), (border, border), (border, -border), (-border, border), 
                      (-border, 0), (border, 0), (0, -border), (0, border)]:
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Draw main text
        draw.text(position, text, font=font, fill=color)
        
        # Convert back to BGR
        image[:] = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    
    def compute_median(self, positions):
        if len(positions) >= 15:
            return statistics.median(positions[-15:])
        return statistics.median(positions)
    
    def compute_smoothed_position(self, medians):
        if len(medians) >= 10:
            return sum(medians[-10:]) / 10
        return sum(medians) / len(medians)
    
    def display_frame(self, frame):
        # Resize for display
        display_frame = cv2.resize(frame, (640, 360))
        
        # Convert to RGB for tkinter
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update canvas
        self.canvas.config(width=pil_image.width, height=pil_image.height)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
    def combine_video_audio(self):
        temp_output = self.output_path
        final_output = os.path.splitext(self.output_path)[0] + "_final.mp4"
        
        try:
            input_video = ffmpeg.input(temp_output)
            input_audio = ffmpeg.input(self.video_path).audio
            
            self.update_status("Combining video with original audio...")
            
            ffmpeg.output(input_video, input_audio, final_output, vcodec='copy', acodec='aac').run(quiet=True, overwrite_output=True)
            
            self.output_path = final_output
            self.update_status(f"Final video saved to: {os.path.basename(final_output)}")
            
            messagebox.showinfo("Success", f"Processing completed. Video saved to:\n{final_output}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to combine video and audio: {str(e)}")

if __name__ == "__main__":
    root = ctk.CTk()
    app = SpeechRecognitionApp(root)
    root.mainloop()