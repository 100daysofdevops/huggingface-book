# Import necessary modules
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, pipeline

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Load the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', max_length=512, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and testing
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

# Set the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Start the training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Create the sentiment-analysis pipeline with the fine-tuned model
classifier = pipeline(task="sentiment-analysis", model="./fine_tuned_model", tokenizer=tokenizer)

# Classify a sample text
text_example = "I am a HUGE fan of action movie"
results = classifier(text_example)

print(results)
