"""
app.py - Main backend for my AI Summarization Project
Created by: [Your Name]
Description: Basic Flask app to upload videos, texts, and images,
             then generate simple summaries (placeholder logic).
"""

import os
import time
import logging
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Setup folders for uploaded files and summaries
UPLOAD_FOLDER = 'uploads'
SUMMARY_FOLDER = 'summaries'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SUMMARY_FOLDER'] = SUMMARY_FOLDER

# Allowed file types
VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Make sure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

# Set up simple logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_allowed_file(filename, allowed_types):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_types

def delete_old_files(minutes=60):
    """Clean up files older than X minutes from uploads and summaries."""
    now = time.time()
    for folder in [UPLOAD_FOLDER, SUMMARY_FOLDER]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                file_age = now - os.path.getmtime(filepath)
                if file_age > minutes * 60:
                    logger.info(f"Removing old file: {filepath}")
                    os.remove(filepath)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# -------- Video summarization routes --------

@app.route('/video')
def video_page():
    # Show the video upload page
    return render_template('video.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    delete_old_files()

    if 'video' not in request.files:
        return render_template('error.html', message='No video uploaded. Please try again.')

    video_file = request.files['video']
    if video_file.filename == '':
        return render_template('error.html', message='You didn\'t select any video.')

    if video_file and is_allowed_file(video_file.filename, VIDEO_EXTENSIONS):
        filename = secure_filename(video_file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(save_path)
        logger.info(f"Video saved: {filename}")

        # Placeholder summary (replace with AI model later)
        summary_text = f"Summary placeholder for video '{filename}'."

        return render_template('result_video.html', filename=filename, summary=summary_text)

    return render_template('error.html', message='Unsupported video format.')

# -------- Text summarization routes --------

@app.route('/text')
def text_page():
    return render_template('text.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    input_text = request.form.get('text_input', '').strip()

    if not input_text:
        return render_template('error.html', message='Please enter some text.')

    # Placeholder summary logic
    summary_text = f"Summary placeholder: {input_text[:75]}..."

    return render_template('result_text.html', original=input_text, summary=summary_text)

# -------- Image summarization routes --------

@app.route('/image')
def image_page():
    return render_template('image.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files.get('image')

    if not image_file or image_file.filename == '':
        return render_template('error.html', message='No image uploaded.')

    if not is_allowed_file(image_file.filename, IMAGE_EXTENSIONS):
        return render_template('error.html', message='Unsupported image format.')

    filename = secure_filename(image_file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(save_path)
    logger.info(f"Image saved: {filename}")

    # Placeholder image summary (replace with OCR + model)
    summary_text = f"Summary placeholder for image '{filename}'."

    return render_template('result_image.html', filename=filename, summary=summary_text)

# Generic error route (if needed)
@app.route('/error')
def error_page():
    return render_template('error.html', message="Oops, something went wrong!")

if __name__ == '__main__':
    print("Starting Flask server at http://127.0.0.1:5000/")
    app.run(debug=True)
