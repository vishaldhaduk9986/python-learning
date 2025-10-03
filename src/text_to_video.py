from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS
from moviepy import ImageSequenceClip, AudioFileClip
import numpy as np
import colorsys
import os

# Configuration
TEXT_SPEED = 24  # frames per second
TEXT_COLOR = (255, 255, 255)  # white text
FONT_PATH = "arial.ttf"  # Path to your .ttf font file
FONT_SIZE = 60  # font size
VIDEO_SIZE = (1280, 720)  # width, height in pixels
START_BG_COLOR = "#000000"  # black
END_BG_COLOR = "#6638f0"    # purple

# Convert HEX to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Interpolate between two colors in HSV space
def interpolate_color(start_color, end_color, progress):
    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)
    start_h, start_s, start_v = colorsys.rgb_to_hsv(*[x/255.0 for x in start_rgb])
    end_h, end_s, end_v = colorsys.rgb_to_hsv(*[x/255.0 for x in end_rgb])
    h = start_h + (end_h - start_h) * progress
    s = start_s + (end_s - start_s) * progress
    v = start_v + (end_v - start_v) * progress
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def create_video_from_text(text_script_path, output_video_path):
    # Read text from file
    with open(text_script_path, "r") as f:
        text = f.read().strip()
    words = text.split()

    # Create speech audio from text
    tts = gTTS(text=text, lang="en")
    audio_file = "temp_speech.mp3"
    tts.save(audio_file)

    # Load audio duration
    audio_clip = AudioFileClip(audio_file)
    audio_duration = audio_clip.duration
    avg_word_duration = audio_duration / len(words)

    # Load font
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    frames = []
    durations = []

    for i, word in enumerate(words):
        # Create image with interpolated background color
        progress = i / max(len(words) - 1, 1)
        bg_color = interpolate_color(START_BG_COLOR, END_BG_COLOR, progress)
        img = Image.new("RGB", VIDEO_SIZE, color=bg_color)

        draw = ImageDraw.Draw(img)
        text_width, text_height = draw.textsize(word, font=font)
        x = (VIDEO_SIZE[0] - text_width) / 2
        y = (VIDEO_SIZE[1] - text_height) / 2
        draw.text((x, y), word, font=font, fill=TEXT_COLOR)

        frames.append(np.array(img))
        durations.append(avg_word_duration)  # seconds per frame

    # Create video clip from frames and durations
    clip = ImageSequenceClip(frames, durations=durations)
    clip = clip.set_audio(audio_clip)
    clip.write_videofile(output_video_path, fps=TEXT_SPEED, codec="libx264")

    # Clean up temp audio
    os.remove(audio_file)

if __name__ == "__main__":
    input_text_file = "input_text.txt"   # Your text script file path
    output_video_file = "output_video.mp4"  # Output video file path
    create_video_from_text(input_text_file, output_video_file)
