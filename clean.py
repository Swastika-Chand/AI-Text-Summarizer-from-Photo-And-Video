import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset.csv")

# Cleaning for transcript (NO stopword removal to preserve context)
def clean_transcript(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove [noise], [music]
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z0-9.,!?\'\s]', '', text)  # Remove special characters except punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

# Cleaning for summary (preserve stopwords and punctuation)
def clean_summary(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z0-9.,!?\'\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply cleaning
df['cleaned_transcript'] = df['transcript'].apply(clean_transcript)
df['cleaned_summary'] = df['summary'].apply(clean_summary)

# Save cleaned file
df.to_csv("cleaned_dataset.csv", index=False)
print("Cleaned dataset saved as 'cleaned_dataset.csv'")
