from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PIL import Image
import torch
import re
import os

# ------------------ Load TrOCR ------------------ #
def load_trocr_model(model_path, device):
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    processor = TrOCRProcessor.from_pretrained(model_path)
    return model, processor

# ------------------ OCR ------------------ #
def extract_text_from_image(model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ------------------ Cleaning ------------------ #
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.]', '', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)  # Remove duplicated words
    return text.strip()

# ------------------ Validity Check ------------------ #
def is_valid_text(text):
    words = text.split()
    return len(text.strip()) > 3 and len(set(w for w in words if len(w) > 2)) >= 3

# ------------------ Grammar Correction ------------------ #
def improve_grammar(text, grammar_pipe):
    prompt = (
        f"Fix the grammar of this sentence without changing its meaning or adding new content: {text}"
    )
    try:
        result = grammar_pipe(prompt, max_length=80, num_beams=4, early_stopping=True)
        return result[0]["generated_text"].strip()
    except Exception:
        return text

# ------------------ Summary Generation ------------------ #
def get_best_summary(text, tokenizer, model, device):
    prompts = [
        f"Summarize the key action in simple words: {text}",
        f"What's the main idea? {text}",
        f"Explain the core concept of this text: {text}",
        f"Rephrase concisely in 10-12 words: {text}",
        f"Make this more concise without changing the meaning: {text}",
        f"Rewrite this as a natural short summary: {text}",
        f"Express this like a short student note: {text}",
        f"Reframe this into a short educational point: {text}"
    ]

    best_summary = ""
    best_score = 0

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
        outputs = model.generate(input_ids, max_length=60, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Score: unique long keywords + shorter is better
        if is_valid_text(summary) and summary.lower() != text.lower():
            words = summary.split()
            keyword_score = len(set(w for w in words if len(w) > 3))
            penalty = max(0, len(words) - 12)  # encourage short summaries
            score = keyword_score - penalty

            if score > best_score:
                best_score = score
                best_summary = summary

    # Fallback
    if not best_summary:
        fallback_prompt = f"Rewrite this in simpler and clearer words: {text}"
        input_ids = tokenizer(fallback_prompt, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        best_summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return best_summary

# ------------------ Full Flow ------------------ #
def summarize_text(cleaned_text, flan_tokenizer, flan_model, grammar_pipe, device):
    original = cleaned_text.strip()

    if not original:
        return {
            "note": "❗ Empty OCR result.",
            "original": "",
            "summary": ""
        }

    corrected = improve_grammar(original, grammar_pipe)

    if len(corrected.split()) <= 4:
        return {
            "note": "⚠️ Very short text. Using grammar-corrected version as summary.",
            "original": original,
            "corrected": corrected,
            "summary": corrected
        }

    if not is_valid_text(corrected):
        return {
            "note": "⚠️ Low-quality or noisy text. Grammar improved but no quality summary possible.",
            "original": original,
            "corrected": corrected,
            "summary": corrected
        }

    summary = get_best_summary(corrected, flan_tokenizer, flan_model, device)

    note = (
        "✅ Summary rewritten successfully using abstraction and keyword filtering."
        if summary.lower() != corrected.lower()
        else "ℹ️ No abstracted summary. Fallback rephrasing applied."
    )

    return {
        "note": note,
        "original": original,
        "corrected": corrected,
        "summary": summary
    }

# ------------------ Main Entry ------------------ #
def main():
    TROCR_MODEL_PATH = r"C:\Users\acer\OneDrive\Desktop\project\trocr_finetuned\trocr_finetuned"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    trocr_model, trocr_processor = load_trocr_model(TROCR_MODEL_PATH, device)
    flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

    grammar_pipe = pipeline(
        "text2text-generation",
        model="pszemraj/flan-t5-large-grammar-synthesis",
        device=0 if torch.cuda.is_available() else -1
    )

    image_path = input("📁 Enter image path: ").strip()
    if not os.path.exists(image_path):
        print("❌ Invalid image path.")
        return

    print("\n📷 Extracting text...")
    raw_text = extract_text_from_image(trocr_model, trocr_processor, image_path, device)
    print("\n📝 Extracted Text:\n", raw_text)

    cleaned = clean_text(raw_text)
    print("\n🧹 Cleaned Text:\n", cleaned)

    print("\n🔍 Generating summary...")
    result = summarize_text(cleaned, flan_tokenizer, flan_model, grammar_pipe, device)

    print("\n Note:", result["note"])
    print("\n Original Text:\n", result["original"])
    if "corrected" in result:
        print("\n Corrected Grammar:\n", result["corrected"])
    print("\n Final Summary:\n", result["summary"])

if __name__ == "__main__":
    main()
