from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS
from moviepy import ImageSequenceClip, AudioFileClip
import numpy as np
import colorsys
import os

# Configuration
TEXT_SPEED = 24  # frames per second
TEXT_COLOR = (255, 255, 255)  # white text
FONT_PATH = "arial.ttf"  # Path to your .ttf font file (will fallback to default if missing)
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

def create_video_from_text(text_script_path, output_video_path, with_cartoon=False):
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

    # Load font with fallback to default bitmap font if the .ttf is missing
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    frames = []
    durations = []

    for i, word in enumerate(words):
        # Create image with interpolated background color
        progress = i / max(len(words) - 1, 1)
        bg_color = interpolate_color(START_BG_COLOR, END_BG_COLOR, progress)
        img = Image.new("RGB", VIDEO_SIZE, color=bg_color)

        draw = ImageDraw.Draw(img)
        if with_cartoon:
            # Draw a simple cartoon character on the left
            head_radius = 60
            head_center = (int(VIDEO_SIZE[0] * 0.18), int(VIDEO_SIZE[1] * 0.5))
            draw.ellipse([
                head_center[0] - head_radius,
                head_center[1] - head_radius,
                head_center[0] + head_radius,
                head_center[1] + head_radius,
            ], fill=(255, 224, 189), outline=(0, 0, 0), width=3)
            # eyes
            eye_offset_x = 22
            eye_offset_y = -8
            eye_size = 8
            draw.ellipse([
                head_center[0] - eye_offset_x - eye_size,
                head_center[1] + eye_offset_y - eye_size,
                head_center[0] - eye_offset_x + eye_size,
                head_center[1] + eye_offset_y + eye_size,
            ], fill=(0, 0, 0))
            draw.ellipse([
                head_center[0] + eye_offset_x - eye_size,
                head_center[1] + eye_offset_y - eye_size,
                head_center[0] + eye_offset_x + eye_size,
                head_center[1] + eye_offset_y + eye_size,
            ], fill=(0, 0, 0))
            # mouth
            mouth_width = 30
            mouth_height = 8 if i % 2 == 0 else 20
            mouth_top = head_center[1] + 26
            draw.ellipse([
                head_center[0] - mouth_width,
                mouth_top - mouth_height,
                head_center[0] + mouth_width,
                mouth_top + mouth_height,
            ], fill=(150, 0, 0), outline=(0, 0, 0))

            # speech bubble on the right
            bubble_x0 = int(VIDEO_SIZE[0] * 0.35)
            bubble_y0 = int(VIDEO_SIZE[1] * 0.25)
            bubble_x1 = int(VIDEO_SIZE[0] * 0.92)
            bubble_y1 = int(VIDEO_SIZE[1] * 0.6)
            draw.rectangle([bubble_x0, bubble_y0, bubble_x1, bubble_y1], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
            # word text inside bubble
            bbox = draw.textbbox((0, 0), word, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            tx = bubble_x0 + (bubble_x1 - bubble_x0 - text_width) / 2
            ty = bubble_y0 + (bubble_y1 - bubble_y0 - text_height) / 2
            draw.text((tx, ty), word, font=font, fill=(0, 0, 0))
        else:
            # Compute text size using textbbox for Pillow compatibility
            bbox = draw.textbbox((0, 0), word, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (VIDEO_SIZE[0] - text_width) / 2
            y = (VIDEO_SIZE[1] - text_height) / 2
            draw.text((x, y), word, font=font, fill=TEXT_COLOR)

        frames.append(np.array(img))
        durations.append(avg_word_duration)  # seconds per frame

    # Create video clip from frames and durations
    clip = ImageSequenceClip(frames, durations=durations)
    # Attach the audio clip in a version-agnostic way
    try:
        clip.audio = audio_clip
    except Exception:
        # Some versions may not allow direct assignment; ignore and rely on writer to merge via file if possible
        pass
    clip.write_videofile(output_video_path, fps=TEXT_SPEED, codec="libx264", audio=True)

    # Clean up temp audio
    os.remove(audio_file)


def create_cartoon_video_from_text(text_script_path, output_video_path):
    """Create a simple cartoon-style video from the input text.

    The function draws a simple cartoon character and a speech bubble for each word,
    animating the mouth and background color across the words. Uses gTTS for audio
    (same as the plain mode).
    """
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
    avg_word_duration = audio_duration / max(len(words), 1)

    # Font fallback
    try:
        font = ImageFont.truetype(FONT_PATH, int(FONT_SIZE * 0.8))
    except Exception:
        font = ImageFont.load_default()

    frames = []
    durations = []

    for i, word in enumerate(words):
        progress = i / max(len(words) - 1, 1)
        bg_color = interpolate_color(START_BG_COLOR, END_BG_COLOR, progress)
        img = Image.new("RGB", VIDEO_SIZE, color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw a simple cartoon character on the left
        head_radius = 80
        head_center = (int(VIDEO_SIZE[0] * 0.2), int(VIDEO_SIZE[1] * 0.5))
        # head
        draw.ellipse(
            [
                head_center[0] - head_radius,
                head_center[1] - head_radius,
                head_center[0] + head_radius,
                head_center[1] + head_radius,
            ],
            fill=(255, 224, 189),
            outline=(0, 0, 0),
            width=4,
        )
        # eyes
        eye_offset_x = 30
        eye_offset_y = -10
        eye_size = 10
        draw.ellipse(
            [
                head_center[0] - eye_offset_x - eye_size,
                head_center[1] + eye_offset_y - eye_size,
                head_center[0] - eye_offset_x + eye_size,
                head_center[1] + eye_offset_y + eye_size,
            ],
            fill=(0, 0, 0),
        )
        draw.ellipse(
            [
                head_center[0] + eye_offset_x - eye_size,
                head_center[1] + eye_offset_y - eye_size,
                head_center[0] + eye_offset_x + eye_size,
                head_center[1] + eye_offset_y + eye_size,
            ],
            fill=(0, 0, 0),
        )

        # mouth animation: open for odd words, closed for even words
        mouth_width = 40
        mouth_height = 10 if i % 2 == 0 else 25
        mouth_top = head_center[1] + 30
        draw.ellipse(
            [
                head_center[0] - mouth_width,
                mouth_top - mouth_height,
                head_center[0] + mouth_width,
                mouth_top + mouth_height,
            ],
            fill=(150, 0, 0) if i % 2 else (120, 40, 40),
            outline=(0, 0, 0),
        )

        # Draw a speech bubble on the right with the word
        bubble_margin = 20
        bubble_x0 = int(VIDEO_SIZE[0] * 0.35)
        bubble_y0 = int(VIDEO_SIZE[1] * 0.3)
        bubble_x1 = int(VIDEO_SIZE[0] * 0.9)
        bubble_y1 = int(VIDEO_SIZE[1] * 0.6)
        # bubble rect with rounded corners (approx)
        draw.rectangle([bubble_x0, bubble_y0, bubble_x1, bubble_y1], fill=(255, 255, 255), outline=(0, 0, 0), width=3)

        # word text centered in bubble
        bbox = draw.textbbox((0, 0), word, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        tx = bubble_x0 + (bubble_x1 - bubble_x0 - text_width) / 2
        ty = bubble_y0 + (bubble_y1 - bubble_y0 - text_height) / 2
        draw.text((tx, ty), word, font=font, fill=(0, 0, 0))

        frames.append(np.array(img))
        durations.append(avg_word_duration)

    # Write video; attach audio similarly to plain mode
    clip = ImageSequenceClip(frames, durations=durations)
    try:
        clip.audio = audio_clip
    except Exception:
        pass
    clip.write_videofile(output_video_path, fps=TEXT_SPEED, codec="libx264", audio=True)

    # cleanup
    os.remove(audio_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a video from text (plain or cartoon)")
    parser.add_argument("--mode", choices=["plain", "cartoon"], default="plain", help="Video style mode")
    parser.add_argument("--input", default="input_text.txt", help="Input text file")
    parser.add_argument("--output", default="output_video.mp4", help="Output video file path")
    parser.add_argument("--with-cartoon", action="store_true", help="Overlay a small cartoon in plain mode")
    args = parser.parse_args()

    if args.mode == "plain":
        create_video_from_text(args.input, args.output, with_cartoon=args.with_cartoon)
    else:
        create_cartoon_video_from_text(args.input, args.output)
