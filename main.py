from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import pytesseract
import torch
import re
import os
import cv2

# ---------- CONFIG ---------- #
TROCR_HANDWRITTEN_PATH = r"C:\Users\acer\OneDrive\Desktop\project\trocr_finetuned\trocr_finetuned"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------- MODEL LOADERS ---------- #
def load_handwritten_model(path, device):
    model = VisionEncoderDecoderModel.from_pretrained(path).to(device)
    processor = TrOCRProcessor.from_pretrained(path)
    return model, processor

def load_summarization_models(device):
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

    flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

    return (bart_tokenizer, bart_model), (flan_tokenizer, flan_model)

# ---------- OCR FUNCTIONS ---------- #
def extract_handwritten_text(model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_digital_text(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.medianBlur(image, 3)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)
    image = cv2.fastNlMeansDenoising(image, h=30)
    return pytesseract.image_to_string(image)

# ---------- SUMMARIZATION ---------- #
def summarize_bart(text, tokenizer, model, device):
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=100,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

def summarize_flan(text, tokenizer, model, device):
    # Step 1: Grammar correction
    grammar_prompt = f"Fix grammar and improve clarity without changing meaning: {text}"
    grammar_inputs = tokenizer(grammar_prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    grammar_ids = model.generate(
        **grammar_inputs, max_new_tokens=100,
        temperature=0.7, top_p=0.9, do_sample=True
    )
    corrected_text = tokenizer.decode(grammar_ids[0], skip_special_tokens=True).strip()

    # Step 2: Summary
    summary_prompt = f"Summarize this in a clear and concise way:\n\"{corrected_text}\""
    summary_inputs = tokenizer(summary_prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    summary_ids = model.generate(
        **summary_inputs, max_new_tokens=60,
        temperature=0.7, top_p=0.9, do_sample=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

# ---------- CLEANING ---------- #
def clean_text(text):
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# ---------- MAIN ---------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (bart_tokenizer, bart_model), (flan_tokenizer, flan_model) = load_summarization_models(device)

    print("Choose image type:\n1. Handwritten\n2. Printed/Digital")
    choice = input("Enter 1 or 2: ").strip()
    image_path = input("Enter full image path: ").strip()

    if not os.path.exists(image_path):
        print("❌ Error: Invalid image path.")
        return

    if choice == "1":
        trocr_model, trocr_processor = load_handwritten_model(TROCR_HANDWRITTEN_PATH, device)
        raw_text = extract_handwritten_text(trocr_model, trocr_processor, image_path, device)
    elif choice == "2":
        raw_text = extract_digital_text(image_path)
    else:
        print("❌ Error: Invalid choice.")
        return

    cleaned_text = clean_text(raw_text)
    print("\n📝 Extracted Text:\n", cleaned_text)

    if len(cleaned_text.split()) <= 25:
        summary = summarize_flan(cleaned_text, flan_tokenizer, flan_model, device)
    else:
        summary = summarize_bart(cleaned_text, bart_tokenizer, bart_model, device)

    print("\n📌 Summary:\n", summary)

if __name__ == "__main__":
    main()