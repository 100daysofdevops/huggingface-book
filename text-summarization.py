from transformers import pipeline

text = (
    "The rise of artificial intelligence (AI) in various industries has been remarkable. "
    "From healthcare to finance, AI is transforming the way we live and work. With advancements in machine learning "
    "and natural language processing, AI systems are becoming more efficient and capable of performing complex tasks. "
    "Companies are investing heavily in AI research and development, aiming to stay ahead in the competitive market."
)

# Create the summarization pipeline
summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum")

# Summarize the text
summary_text = summarizer(text, min_length=30, max_length=60)

# Compare the length
print(f"Original text length: {len(text)}")
print(f"Summary length: {len(summary_text[0]['summary_text'])}")
