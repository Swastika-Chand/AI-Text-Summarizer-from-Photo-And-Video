from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd

# Load cleaned data
df = pd.read_csv("cleaned_dataset.csv")

# Prepare HuggingFace dataset
dataset = Dataset.from_pandas(
    df[['cleaned_transcript', 'cleaned_summary']]
    .rename(columns={'cleaned_transcript': 'text', 'cleaned_summary': 'summary'})
)

# Train-test split
dataset = dataset.train_test_split(test_size=0.1)

# Load tokenizer and model (use large cnn version for better performance)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Tokenize function
def tokenize(batch):
    inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch['summary'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = labels['input_ids']
    return inputs

# Tokenize dataset
tokenized_data = dataset.map(tokenize, batched=True, remove_columns=["text", "summary"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=4,  # Increase batch size if GPU allows
    per_device_eval_batch_size=4,
    learning_rate=3e-5,
    num_train_epochs=5,  # More epochs for better convergence
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=150,
    generation_num_beams=6,  # Use bigger beam size during generation
)

# Define Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save model
model.save_pretrained("./summarization_model")
tokenizer.save_pretrained("./summarization_model")
