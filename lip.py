from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from transformers import pipeline
import pandas as pd
from urllib.parse import urlparse, parse_qs
from xml.etree.ElementTree import ParseError

# Initialize summarizer (use GPU if available, else remove device=0)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

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
    except ParseError as e:
        return None, f"XML Parse Error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# Summarize entire transcript
def summarize_transcript(transcript):
    try:
        # Hugging Face models have max input limits (~1024 tokens), so if transcript is too long, truncate it
        max_input_words = 450  # roughly corresponds to model max tokens (~1024 tokens)
        words = transcript.split()
        if len(words) > max_input_words:
            print(f"Transcript too long ({len(words)} words), truncating to {max_input_words} words.")
            transcript = " ".join(words[:max_input_words])

        summary = summarizer(transcript, max_length=130, min_length=50, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Warning: summarization failed: {e}")
        return "Summary not available"

# Read URLs from file
with open("video_urls.txt", "r") as file:
    urls = [line.strip() for line in file.readlines()]

data = []
for url in urls:
    print(f"Processing: {url}")
    transcript, error = get_video_transcript(url)
    if error:
        print(f"Skipping: {error}")
        continue
    summary = summarize_transcript(transcript)
    data.append({"url": url, "transcript": transcript, "summary": summary})

df = pd.DataFrame(data)
df.to_csv("dataset.csv", index=False)
print("Dataset created and saved as dataset.csv")
