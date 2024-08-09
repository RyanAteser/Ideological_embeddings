import os
import numpy as np
import csv
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split

# Check if NumPy is installed
try:
    print("NumPy version:", np.__version__)
except ImportError:
    print("NumPy is not installed. Please install it using 'pip install numpy'.")

# Define the corpus without predefined ideologies
corpus = [
    "The Israeli settlements in the West Bank are essential for Israel's security.",
    "Palestinians have the right to return to their homeland.",
    "Israel's military actions are justified to defend its borders.",
    "The international community should recognize Palestinian statehood.",
    "Israeli government policies are crucial for maintaining peace in the region.",
    "The blockade on Gaza should be lifted to allow humanitarian aid.",
    "Israel's right to exist as a Jewish state must be protected.",
    "The occupation of Palestinian territories is a violation of international law.",
    "Support for Israel is necessary to maintain stability in the Middle East.",
    "Palestinian resistance is a legitimate struggle for freedom.",
    "Israeli airstrikes in Gaza are a necessary response to rocket attacks.",
    "The expansion of Israeli settlements undermines the peace process.",
    "Palestinians should have full sovereignty over East Jerusalem.",
    "The United States should continue its military aid to Israel.",
    "The humanitarian crisis in Gaza is a direct result of Israeli policies.",
    "Israel's defense strategies are necessary to combat terrorism.",
    "The international community should impose sanctions on Israel for its actions.",
    "Jerusalem should remain the undivided capital of Israel.",
    "Palestinian political movements have a right to self-determination.",
    "Israeli security forces act in self-defense during confrontations.",
    "Palestinian protests against the occupation are a form of civil resistance.",
    "Israel's technological innovations contribute to global security.",
    "The siege on Gaza is an unjust act of collective punishment.",
    "Israel's partnerships with neighboring countries are key to regional stability.",
    "The right of return for Palestinian refugees is non-negotiable.",
    "Both sides need to come together to find a peaceful resolution.",
    "The conflict has caused suffering on both sides.",
    "Peace talks are essential for long-term stability.",
    "There needs to be a mutual understanding to resolve the issues.",
    "Violence from both sides only perpetuates the cycle of conflict."
]

# Define the ideologies for the benchmark
benchmark_data = [
    {"text": "The Israeli settlements in the West Bank are essential for Israel's security.", "ideology": "Pro-Israeli"},
    {"text": "Palestinians have the right to return to their homeland.", "ideology": "Pro-Palestine"},
    {"text": "Israel's military actions are justified to defend its borders.", "ideology": "Pro-Israeli"},
    {"text": "The international community should recognize Palestinian statehood.", "ideology": "Pro-Palestine"},
    {"text": "Israeli government policies are crucial for maintaining peace in the region.", "ideology": "Pro-Israeli"},
    {"text": "The blockade on Gaza should be lifted to allow humanitarian aid.", "ideology": "Pro-Palestine"},
    {"text": "Israel's right to exist as a Jewish state must be protected.", "ideology": "Pro-Israeli"},
    {"text": "The occupation of Palestinian territories is a violation of international law.", "ideology": "Pro-Palestine"},
    {"text": "Support for Israel is necessary to maintain stability in the Middle East.", "ideology": "Pro-Israeli"},
    {"text": "Palestinian resistance is a legitimate struggle for freedom.", "ideology": "Pro-Palestine"},
    {"text": "Israeli airstrikes in Gaza are a necessary response to rocket attacks.", "ideology": "Pro-Israeli"},
    {"text": "The expansion of Israeli settlements undermines the peace process.", "ideology": "Pro-Palestine"},
    {"text": "Palestinians should have full sovereignty over East Jerusalem.", "ideology": "Pro-Palestine"},
    {"text": "The United States should continue its military aid to Israel.", "ideology": "Pro-Israeli"},
    {"text": "The humanitarian crisis in Gaza is a direct result of Israeli policies.", "ideology": "Pro-Palestine"},
    {"text": "Israel's defense strategies are necessary to combat terrorism.", "ideology": "Pro-Israeli"},
    {"text": "The international community should impose sanctions on Israel for its actions.", "ideology": "Pro-Palestine"},
    {"text": "Jerusalem should remain the undivided capital of Israel.", "ideology": "Pro-Israeli"},
    {"text": "Palestinian political movements have a right to self-determination.", "ideology": "Pro-Palestine"},
    {"text": "Israeli security forces act in self-defense during confrontations.", "ideology": "Pro-Israeli"},
    {"text": "Palestinian protests against the occupation are a form of civil resistance.", "ideology": "Pro-Palestine"},
    {"text": "Israel's technological innovations contribute to global security.", "ideology": "Pro-Israeli"},
    {"text": "The siege on Gaza is an unjust act of collective punishment.", "ideology": "Pro-Palestine"},
    {"text": "Israel's partnerships with neighboring countries are key to regional stability.", "ideology": "Pro-Israeli"},
    {"text": "The right of return for Palestinian refugees is non-negotiable.", "ideology": "Pro-Palestine"},
    {"text": "Both sides need to come together to find a peaceful resolution.", "ideology": "Neutral"},
    {"text": "The conflict has caused suffering on both sides.", "ideology": "Neutral"},
    {"text": "Peace talks are essential for long-term stability.", "ideology": "Neutral"},
    {"text": "There needs to be a mutual understanding to resolve the issues.", "ideology": "Neutral"},
    {"text": "Violence from both sides only perpetuates the cycle of conflict.", "ideology": "Neutral"}
]

# Convert to DataFrame and save to CSV
df = pd.DataFrame({"text": corpus})
df.to_csv('news_articles.csv', index=False)
print("Data collection and annotation completed.")

# Convert benchmark data to DataFrame and save to CSV
benchmark_df = pd.DataFrame(benchmark_data, columns=["text", "ideology"])
benchmark_df.to_csv('benchmark_data.csv', index=False)
print("Benchmark data saved.")

# Load pre-trained BERT model and tokenizer from Hugging Face
print("Loading BERT model and tokenizer...")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 ideologies: Pro-Israeli, Pro-Palestine, Neutral

# Create a custom Dataset for fine-tuning
class IdeologyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dimension
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

# Define the labels for training
labels = [0 if "Pro-Israeli" in text else 1 if "Pro-Palestine" in text else 2 for text in df["text"]]

# Split data into training and evaluation sets
train_texts, eval_texts, train_labels, eval_labels = train_test_split(corpus, labels, test_size=0.2, random_state=42)

train_dataset = IdeologyDataset(train_texts, train_labels, tokenizer)
eval_dataset = IdeologyDataset(eval_texts, eval_labels, tokenizer)

# Fine-tune the BERT model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed.")

# Save the fine-tuned model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
print("Model and tokenizer saved.")

# Function to identify the ideology from a given text using BERT
def identify_ideology_bert(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    ideologies = ["Pro-Israeli", "Pro-Palestine", "Neutral"]
    return logits, predicted_class_id, ideologies[predicted_class_id]

# Function to check if a new ideological position is novel
def is_novel(position, archive, threshold):
    position_array = np.array(position, dtype=float)
    for archived_array in archive:
        archived_array = np.array(archived_array, dtype=float)
        if np.linalg.norm(position_array - archived_array) < threshold:
            return False
    return True

# Intelligent Go-Explore inspired components
archive = defaultdict(list)
exploration_threshold = 0.5

# Training process with exploration
epochs = 10
learning_rate = 0.01

print("Starting training process with exploration...")
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i, text in enumerate(corpus):
        logits, predicted_class_id, ideology_label = identify_ideology_bert(text)
        if is_novel(logits.detach().numpy(), archive[ideology_label], exploration_threshold):
            archive[ideology_label].append(logits.detach().numpy())
        print(f"Epoch {epoch + 1}/{epochs}, Text: {text[:50]}, Predicted Ideology: {ideology_label}, Alignment Probability: {torch.nn.functional.softmax(logits, dim=1).max().item()}")

print("Exploration and training completed.")

# Save benchmark data to CSV file
benchmark_df.to_csv('benchmark_data.csv', index=False)
print("Benchmark data saved.")

# Function to evaluate benchmark data from a CSV file
def evaluate_benchmark_from_csv(csv_file_path):
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row['text']
            expected_ideology = row['ideology']
            _, _, predicted_ideology = identify_ideology_bert(text)
            print(f"Text: {text}")
            print(f"Expected Ideology: {expected_ideology}")
            print(f"Predicted Ideology: {predicted_ideology}")
            print("Match" if expected_ideology == predicted_ideology else "Mismatch")
            print("")

# Evaluate the benchmark using the CSV file
evaluate_benchmark_from_csv('benchmark_data.csv')

if __name__ == "__main__":
    text = input("Enter a text to identify its ideology: ")
    _, _, ideology = identify_ideology_bert(text)
    print(f"\nIdeology of the given text: {ideology}")
