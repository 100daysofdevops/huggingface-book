from transformers import pipeline

# Define the summarization pipeline
summarizer = pipeline("summarization", truncation=True)

# List of texts
texts = [
    "The rise of artificial intelligence (AI) in various industries has been remarkable. "
    "From healthcare to finance, AI is transforming the way we live and work. With advancements in machine learning "
    "and natural language processing, AI systems are becoming more efficient and capable of performing complex tasks. "
    "Companies are investing heavily in AI research and development, aiming to stay ahead in the competitive market.",
    
    "Climate change is one of the most pressing issues of our time. The increasing levels of greenhouse gases in the atmosphere "
    "are causing global temperatures to rise, leading to more frequent and severe weather events. Governments and organizations "
    "worldwide are working to mitigate the impact of climate change by reducing emissions and promoting sustainable practices."
]

# Summarize the texts
summaries = [summarizer(text, min_length=30, max_length=60) for text in texts]

# Print the summaries
for summary in summaries:
    print(summary)
