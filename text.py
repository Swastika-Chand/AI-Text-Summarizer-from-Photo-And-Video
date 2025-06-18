from transformers import pipeline

def summarize_text_pegasus(text, max_length=150, min_length=40):
    """
    Uses a pre-trained PEGASUS model (google/pegasus-xsum) to summarize the given text.
    """
    try:
        # Load the summarization pipeline with the PEGASUS model
        # 'google/pegasus-xsum' is a good choice for highly abstractive summaries
        summarizer = pipeline("summarization", model="google/pegasus-xsum")

        # Generate the summary
        # PEGASUS doesn't require a special prefix like T5
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False, # Set to False for greedy/beam search
            num_beams=4,     # Use beam search for better quality
            early_stopping=True
        )[0]['summary_text']
        return summary
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("--- Text Summarization AI Project (using PEGASUS) ---")
    print("Enter text to summarize (type 'quit' or 'exit' to stop):")

    while True:
        user_input = input("\nEnter your text: \n")
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting summarizer. Goodbye!")
            break
        if not user_input.strip():
            print("Please enter some text.")
            continue

        print("\nSummarizing...")
        # Adjust max_length and min_length as needed for your desired summary length
        generated_summary = summarize_text_pegasus(user_input, max_length=130, min_length=30)
        print("\n--- Generated Summary ---")
        print(generated_summary)
        print("-------------------------\n")