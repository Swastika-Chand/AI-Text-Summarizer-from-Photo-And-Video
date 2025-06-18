from fastapi import FastAPI
from transformers import BartTokenizer, BartForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
import re
import torch

app = FastAPI()

# Load model and tokenizer (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = BartForConditionalGeneration.from_pretrained("./summarization_model").to(device)
    tokenizer = BartTokenizer.from_pretrained("./summarization_model")
except Exception as e:
    raise RuntimeError(f"Failed to load model/tokenizer: {e}")

# Clean transcript text helper
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Summarize function
def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)
    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        num_beams=6,
        max_length=150,
        early_stopping=True,
        length_penalty=2.0,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Extract YouTube video ID from URL 
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None

# Get YouTube transcript text
def get_video_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return None, "Invalid YouTube URL"
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript]), None
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return None, f"Transcript unavailable: {str(e)}"

# Input schema for POST requests
class VideoURL(BaseModel):
    url: str

@app.post("/summarize/")
async def summarize_video(video: VideoURL):
    transcript, error = get_video_transcript(video.url)
    if error:
        return {"error": error}
    cleaned_transcript = clean_text(transcript)
    summary = summarize_text(cleaned_transcript)
    return {"summary": summary}