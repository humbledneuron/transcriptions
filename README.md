# Real-Time Speech Recognition and Subtitling Tool
This project extracts audio from a video, recognizes speech using Vosk, and overlays the transcriptions onto the video. It also detects faces in the video to align subtitles close to face positions and handles both English and Japanese text overlays.

If this software is useful to you, please consider subscribing to my Youtube Channel to help support this and other open-source projects: https://www.youtube.com/@jeoresearch

The output of this tool should roughly look like the following video: https://youtu.be/9bWzUjcxZWg

## Requirements

Make sure you have the following dependencies installed before running the code:

- [Vosk](https://github.com/alphacep/vosk-api) (Speech recognition model)
- OpenCV (`cv2`)
- Pillow (`PIL`)
- ffmpeg (`ffmpeg-python`)
- sounddevice
- numpy
- statistics
- threading
- queue
- json
- os
- time

You can install most of these dependencies using pip:

```bash
pip install vosk opencv-python Pillow ffmpeg-python sounddevice numpy
```

Install FFmpeg: Download and install FFmpeg from [here](https://ffmpeg.org/download.html). Make sure it's added to your system's PATH.

Download Vosk Model:

Download the Vosk models from [here](https://alphacephei.com/vosk/models).
Extract the model and update the model_path variable in the script with the correct path.
Directory Paths to Update:
Vosk Model Path: Set the path to the downloaded Vosk model:

```python
model_path = "C:/path_to_your_vosk_model/vosk-model-en-us-0.22-lgraph"
```

Font Path (if using Japanese text): Update the font_path in the put_japanese_text function to point to your local fonts.

```python
font_path = 'E:/Software/JPFonts/NotoSansJP-Regular.ttf'
```

Video File Paths:

Input Video: Update the video_paths to point to the video file you want to process:
video_path = "C:/path_to_your_video/input_video.mp4"
output_video_path = 'C:/path_to_your_output/output_with_subtitles.mp4'

Prepare the Vosk Model: Download and extract the model for your preferred language, then set the model_path to the correct directory.

Run the Script: After setting up the directories, run the script with:

```bash
python script_name.py
```


