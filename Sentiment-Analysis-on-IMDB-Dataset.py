from datasets import load_dataset
from transformers import pipeline

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze sentiment for a few sample reviews
for i in range(3):
    sample_review = dataset["train"][i]["text"]
    result = sentiment_analyzer(sample_review)
    print(f"Review: {sample_review}")
    print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.2f}")
    print("-" * 50)
