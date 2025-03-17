import time, json, queue, threading, vosk, sounddevice as sd, cv2, numpy as np, os, statistics, ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageTk
import customtkinter as ctk
from customtkinter import ThemeManager

class SpeechRecognitionApp:
    def __init__(self):
        self.root = ctk.CTk()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root.title("Smart Subtitle Generator")
        self.root.geometry("1280x800")
        self.root.minsize(1280, 720)
        self.setup_vars()
        self.create_ui()
        
    def setup_vars(self):
        self.video_path, self.output_path, self.model_path = "", "", ""
        self.is_recording = self.is_processing = self.is_live = False
        self.language = ctk.StringVar(value="English")
        self.results, self.face_centers_x, self.face_centers_y = [], [], []
        self.face_sizes_y, self.median_x_values, self.median_y_values = [], [], []
        self.audio_queue = queue.Queue()
        self.transcriptions, self.partial_transcription = ["", "", ""], [""]
        self.lock, self.timer, self.prev_length_difference = threading.Lock(), 0, 0
        self.text_scale, self.thickness = 3.0, 4
        self.subtitle_color, self.outline_color = (0, 0, 255), (255, 255, 255)
    
    def create_ui(self):
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        sidebar = ctk.CTkFrame(self.root, width=320, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        
        main_area = ctk.CTkFrame(self.root, corner_radius=0, fg_color=ThemeManager.theme["CTkFrame"]["fg_color"])
        main_area.grid(row=0, column=1, sticky="nsew")
        
        title_label = ctk.CTkLabel(sidebar, text="Smart Subtitle Generator", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(20, 30))
        
        section_frame = ctk.CTkFrame(sidebar)
        section_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(section_frame, text="Language", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")
        language_menu = ctk.CTkSegmentedButton(section_frame, values=["English", "Telugu"], variable=self.language)
        language_menu.pack(fill="x", pady=(5, 0))
        
        section_frame = ctk.CTkFrame(sidebar)
        section_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(section_frame, text="Model", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")
        model_button = ctk.CTkButton(section_frame, text="Select Speech Model", command=self.select_model, 
                               height=40, font=ctk.CTkFont(size=14), corner_radius=8)
        model_button.pack(fill="x", pady=(5, 0))
        self.model_label = ctk.CTkLabel(section_frame, text="No model selected", text_color="gray")
        self.model_label.pack(anchor="w", pady=(5, 0))
        
        section_frame = ctk.CTkFrame(sidebar)
        section_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(section_frame, text="Input Source", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")
        
        source_frame = ctk.CTkFrame(section_frame)
        source_frame.pack(fill="x", pady=(5, 0))
        source_frame.grid_columnconfigure(0, weight=1)
        source_frame.grid_columnconfigure(1, weight=1)
        
        live_btn = ctk.CTkButton(source_frame, text="Camera", command=lambda: self.set_input_mode(True),
                              height=40, corner_radius=8, border_width=0)
        live_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        file_btn = ctk.CTkButton(source_frame, text="Video File", command=lambda: self.set_input_mode(False),
                              height=40, corner_radius=8, border_width=0)
        file_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.file_frame = ctk.CTkFrame(section_frame)
        
        file_select = ctk.CTkButton(self.file_frame, text="Select Video", command=self.select_video, 
                            height=40, corner_radius=8)
        file_select.pack(fill="x", pady=(5, 0))
        
        self.video_label = ctk.CTkLabel(self.file_frame, text="No video selected", text_color="gray")
        self.video_label.pack(anchor="w", pady=(5, 0))
        
        output_select = ctk.CTkButton(self.file_frame, text="Set Output Location", command=self.set_output, 
                               height=40, corner_radius=8)
        output_select.pack(fill="x", pady=(10, 0))
        
        self.output_label = ctk.CTkLabel(self.file_frame, text="Default output location", text_color="gray")
        self.output_label.pack(anchor="w", pady=(5, 0))
        
        section_frame = ctk.CTkFrame(sidebar)
        section_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(section_frame, text="Appearance", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")
        
        appearance_frame = ctk.CTkFrame(section_frame)
        appearance_frame.pack(fill="x", pady=(5, 0))
        appearance_frame.grid_columnconfigure(0, weight=1)
        appearance_frame.grid_columnconfigure(1, weight=4)
        
        ctk.CTkLabel(appearance_frame, text="Size:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        size_slider = ctk.CTkSlider(appearance_frame, from_=1.0, to=5.0, number_of_steps=8,
                              command=lambda v: setattr(self, 'text_scale', v))
        size_slider.set(3.0)
        size_slider.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(appearance_frame, text="Color:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        colors_frame = ctk.CTkFrame(appearance_frame)
        colors_frame.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        color_options = [("Red", (0,0,255)), ("Blue", (255,0,0)), ("Green", (0,255,0)), 
                      ("Yellow", (0,255,255)), ("White", (255,255,255))]
        
        for i, (name, color_value) in enumerate(color_options):
            color_btn = ctk.CTkButton(colors_frame, text="", width=20, height=20, 
                                fg_color=self.rgb_to_hex(color_value), 
                                command=lambda c=color_value: setattr(self, 'subtitle_color', c),
                                corner_radius=10)
            color_btn.pack(side="left", padx=5)
        
        control_frame = ctk.CTkFrame(sidebar)
        control_frame.pack(fill="x", padx=20, pady=(30, 20))
        
        self.start_btn = ctk.CTkButton(control_frame, text="Start Processing", command=self.start_processing,
                              height=50, font=ctk.CTkFont(size=16, weight="bold"), 
                              fg_color="#28a745", hover_color="#218838", corner_radius=8)
        self.start_btn.pack(fill="x", pady=5)
        
        self.stop_btn = ctk.CTkButton(control_frame, text="Stop", command=self.stop_processing,
                             height=50, font=ctk.CTkFont(size=16, weight="bold"), 
                             fg_color="#dc3545", hover_color="#c82333", corner_radius=8,
                             state="disabled")
        self.stop_btn.pack(fill="x", pady=5)
        
        self.canvas_frame = ctk.CTkFrame(main_area, corner_radius=0)
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        status_frame = ctk.CTkFrame(main_area, height=30, corner_radius=0, fg_color=ThemeManager.theme["CTkFrame"]["fg_color"])
        status_frame.pack(fill="x", side="bottom")
        
        self.status_var = ctk.StringVar(value="Ready")
        status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var, anchor="w")
        status_label.pack(side="left", padx=20)
        
        self.set_input_mode(False)
    
    def rgb_to_hex(self, rgb):
        return f"#{rgb[2]:02x}{rgb[1]:02x}{rgb[0]:02x}"
    
    def set_input_mode(self, is_live):
        self.is_live = is_live
        if is_live:
            self.file_frame.pack_forget()
        else:
            self.file_frame.pack(fill="x", pady=(5, 0))
    
    def select_model(self):
        model_dir = ctk.filedialog.askdirectory(title="Select Vosk Model Directory")
        if model_dir:
            self.model_path = model_dir
            self.model_label.configure(text=os.path.basename(model_dir), text_color="white")
            self.update_status(f"Model: {os.path.basename(model_dir)}")
    
    def select_video(self):
        video_path = ctk.filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv;*.mov")])
        if video_path:
            self.video_path = video_path
            self.video_label.configure(text=os.path.basename(video_path), text_color="white")
            if not self.output_path:
                default_name = f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                self.output_path = os.path.join(os.path.dirname(video_path), default_name)
                self.output_label.configure(text=os.path.basename(self.output_path), text_color="white")
            self.update_status(f"Video: {os.path.basename(video_path)}")
            try:
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if ret:
                    self.display_frame(frame)
                cap.release()
            except Exception as e:
                pass
    
    def set_output(self):
        output_path = ctk.filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if output_path:
            self.output_path = output_path
            self.output_label.configure(text=os.path.basename(output_path), text_color="white")
            self.update_status(f"Output: {os.path.basename(output_path)}")
    
    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def start_processing(self):
        if self.is_live and not self.model_path:
            ctk.CTkMessagebox(title="Error", message="Please select a model first")
            return
        
        if not self.is_live and (not self.video_path or not self.model_path):
            ctk.CTkMessagebox(title="Error", message="Please select both a video and a model")
            return
        
        if not self.output_path and not self.is_live:
            default_name = f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            self.output_path = os.path.join(os.path.dirname(self.video_path), default_name)
            self.output_label.configure(text=os.path.basename(self.output_path), text_color="white")
        
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        threading.Thread(target=self.process_live if self.is_live else self.process_video, daemon=True).start()
    
    def stop_processing(self):
        self.is_processing = False
        self.is_recording = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.update_status("Processing stopped")
    
    def process_live(self):
        self.is_processing = self.is_recording = True
        self.results = []
        
        try:
            model = vosk.Model(self.model_path)
            device_info = sd.query_devices(None, 'input')
            samplerate = int(device_info['default_samplerate'])
            
            def callback(indata, frames, time, status):
                if self.is_recording:
                    self.audio_queue.put(bytes(indata))
            
            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16', channels=1, callback=callback):
                self.update_status("Recording started")
                threading.Thread(target=self.recognize_speech, args=(model, samplerate, 30), daemon=True).start()
                
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    ctk.CTkMessagebox(title="Error", message="Could not open webcam")
                    self.stop_processing()
                    return
                
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                frame_idx = 0
                
                while self.is_processing:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    self.process_frame(frame, face_cascade, frame_idx)
                    frame_idx += 1
                    self.display_frame(frame)
                    
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                
                cap.release()
                self.is_recording = False
                self.update_status("Live processing stopped")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.stop_processing()
    
    def process_video(self):
        self.is_processing = True
        self.results = []
        
        try:
            model = vosk.Model(self.model_path)
            threading.Thread(target=self.extract_audio, daemon=True).start()
            threading.Thread(target=self.recognize_speech, args=(model,), daemon=True).start()
            
            self.update_status("Processing video...")
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                ctk.CTkMessagebox(title="Error", message="Could not open video file")
                self.stop_processing()
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            frame_idx = 0
            
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.process_frame(frame, face_cascade, frame_idx)
                frame_idx += 1
                out.write(frame)
                self.display_frame(frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            cap.release()
            out.release()
            
            if self.is_processing:
                self.combine_video_audio()
            
            self.is_processing = False
            self.update_status("Video processing completed")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.stop_processing()
    
    def extract_audio(self):
        process = (ffmpeg.input(self.video_path).output('pipe:', format='wav', ac=1, ar='16k').run_async(pipe_stdout=True))
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
                        transcription = self.process_text(result['text'])
                        self.results.append((frame_number, 'full', transcription))
                else:
                    partial = json.loads(rec.PartialResult())
                    if partial['partial']:
                        self.results.append((frame_number, 'partial', partial['partial']))
            except queue.Empty:
                continue
    
    def process_text(self, text):
        words = text.split()
        if not words:
            return text
        if words[0].lower() == "the":
            words = words[1:]
        if words and words[-1].lower() == "the":
            words = words[:-1]
        
        processed = ' '.join(words)
        if processed:
            processed = processed[0].upper() + processed[1:]
        if processed and processed[-1] not in ".!?":
            processed += "."
        
        return processed
    
    def process_frame(self, frame, face_cascade, frame_idx):
        if self.timer > 0:
            self.timer -= 2
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 0.25
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            if w > 40:
                face_scale = int(1/scale)
                x, y, w, h = x*face_scale, y*face_scale, w*face_scale, h*face_scale
                
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
        
        x0, y0, chin_y = frame.shape[1]//2, 10, 130
        
        if self.face_centers_x and self.face_centers_y:
            median_x = statistics.median(self.face_centers_x[-15:] if len(self.face_centers_x) >= 15 else self.face_centers_x)
            median_y = statistics.median(self.face_centers_y[-15:] if len(self.face_centers_y) >= 15 else self.face_centers_y)
            chin_y = statistics.median(self.face_sizes_y[-15:] if len(self.face_sizes_y) >= 15 else self.face_sizes_y)
            
            self.median_x_values.append(median_x)
            self.median_y_values.append(median_y)
            
            x0 = sum(self.median_x_values[-10:]) / min(10, len(self.median_x_values))
            y0 = sum(self.median_y_values[-10:]) / min(10, len(self.median_y_values))
        
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
        
        dy = 20
        with self.lock:
            lines, length_diff = self.render_subtitles(frame, self.transcriptions, self.partial_transcription,
                                                 x0, y0 + chin_y + 30 + self.timer, dy)
        
        if length_diff - self.prev_length_difference == 1:
            self.timer += dy * 2.5
        
        self.prev_length_difference = length_diff
    
    def render_subtitles(self, frame, transcriptions, partial, x0, y0, dy):
        lines = []
        max_length = 25 if self.language.get() == "Telugu" else 40
        
        combined = ' '.join(transcriptions) + self.process_text(' '.join(partial))
        words = combined.split()
        
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_length:
                current_line = f"{current_line} {word}" if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        original_len = len(lines)
        lines = lines[-3:] 
        length_diff = original_len - len(lines)
        
        height, width = frame.shape[:2]
        
        for i, line in enumerate(lines):
            y = int(y0 + i * dy * self.text_scale)
            (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, 1)
            x = int(x0 - text_width // 2)
            
            border = 2
            for dx, dy in [(-border, -border), (border, border), (border, -border), (-border, border),
                          (-border, 0), (border, 0), (0, -border), (0, border)]:
                cv2.putText(frame, line, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5 * self.text_scale, self.outline_color, self.thickness, cv2.LINE_AA)
            
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5 * self.text_scale, self.subtitle_color, self.thickness-1, cv2.LINE_AA)
        
        return lines, length_diff
    
    def display_frame(self, frame):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 640, 360
        
        frame_h, frame_w = frame.shape[:2]
        scale = min(canvas_width/frame_w, canvas_height/frame_h)
        new_w, new_h = int(frame_w*scale), int(frame_h*scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=img)
        
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2 - new_w//2, canvas_height//2 - new_h//2, 
                              image=self.photo, anchor="nw")
    
    def combine_video_audio(self):
        try:
            final_output = os.path.splitext(self.output_path)[0] + "_final.mp4"
            input_video = ffmpeg.input(self.output_path)
            input_audio = ffmpeg.input(self.video_path).audio
            self.update_status("Finalizing video...")
            ffmpeg.output(input_video, input_audio, final_output, vcodec='copy', acodec='aac').run(quiet=True, overwrite_output=True)
            self.output_path = final_output
            self.update_status(f"Video saved: {os.path.basename(final_output)}")
            ctk.CTkMessagebox(title="Success", message=f"Video saved to: {final_output}")
        except Exception as e:
            ctk.CTkMessagebox(title="Error", message=f"Failed to combine audio: {str(e)}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SpeechRecognitionApp()
    app.run()