import os
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Check if NumPy is installed
try:
    print("NumPy version:", np.__version__)
except ImportError:
    print("NumPy is not installed. Please install it using 'pip install numpy'.")

# Function to scrape news articles
def scrape_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')
    return [article.get_text() for article in articles]

# List of news websites and their ideological leanings
news_sources = {
    'Fox News': 'Conservatism',
    'CNN': 'Progressivism',
    'BBC': 'Moderate'
}

# Example URLs (replace with actual URLs of the news sources)
urls = {
    'Fox News': 'https://www.foxnews.com/',
    'CNN': 'https://www.cnn.com/',
    'BBC': 'https://www.bbc.com/'
}

# Collect and annotate data
data = []
for source, url in urls.items():
    articles = scrape_news(url)
    for article in articles:
        data.append({'text': article, 'ideology': news_sources[source]})

# Convert to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('news_articles.csv', index=False)
print("Data collection and annotation completed.")

# Load the collected corpus from the CSV file
df = pd.read_csv('news_articles.csv')
corpus = df['text'].tolist()
ideologies = df['ideology'].tolist()

# Define the number of topics (K) based on the number of unique ideologies
K = len(set(ideologies))

# Vectorize the corpus and fit LDA to identify topics
print("Vectorizing the corpus and fitting LDA...")
vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2, max_features=1000)
X = vectorizer.fit_transform(corpus)
lda = LatentDirichletAllocation(n_components=K, random_state=0, learning_decay=0.7)
X_topics = lda.fit_transform(X)
topic_words = lda.components_

# Display the topic words for each topic
print("Topic words per topic:")
dynamic_ideology_labels = {}
for i, topic_dist in enumerate(topic_words):
    topic_words_list = [vectorizer.get_feature_names_out()[j] for j in topic_dist.argsort()[:-10 - 1:-1]]
    dynamic_ideology_labels[i] = ', '.join(topic_words_list)
    print(f"Topic {i}: {dynamic_ideology_labels[i]}")

# Manually add ideologies with descriptions
manual_ideologies = {
    0: 'Conservatism',
    1: 'Progressivism',
    2: 'Moderate'
}

# Print manually added ideologies
print("Manually added ideologies:")
for key, value in manual_ideologies.items():
    print(f"Ideology class {key}: {value}")

# Load pre-trained BERT model and tokenizer from Hugging Face
print("Loading BERT model and tokenizer...")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=K)

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

# Map ideologies to integers
ideology_map = {ideology: idx for idx, ideology in enumerate(set(ideologies))}
labels = [ideology_map[ideology] for ideology in ideologies]

# Split data into training and evaluation sets
train_texts, train_labels = corpus[:int(0.8 * len(corpus))], labels[:int(0.8 * len(labels))]
eval_texts, eval_labels = corpus[int(0.8 * len(corpus)):], labels[int(0.8 * len(labels)):]

train_dataset = IdeologyDataset(train_texts, train_labels, tokenizer)
eval_dataset = IdeologyDataset(eval_texts, eval_labels, tokenizer)

# Fine-tune the BERT model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
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
    print(f"Identifying ideology for text: {text[:50]}...")
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    ideology = manual_ideologies[predicted_class_id]
    print(f"Identified ideology class: {predicted_class_id} ({ideology})")
    return logits, predicted_class_id

# Function to ask "What is the ideology?" based on a transcript
def ask_ideology(transcript):
    logits, predicted_class_id = identify_ideology_bert(transcript)
    ideology_label = manual_ideologies[predicted_class_id]
    return ideology_label

# Save benchmark data to CSV file
csv_content = [
    ["Text", "Expected Ideology"],
    ["Support for Israeli settlements in the West Bank is crucial for security.", "Conservatism"],
    ["Palestinian statehood should be recognized and supported by the international community.", "Progressivism"],
    ["Economic cooperation between Israel and Palestine can lead to peace.", "Moderate"],
    ["Military action is necessary to protect Israeli borders from threats.", "Conservatism"],
    ["Human rights abuses against Palestinians must be addressed by global organizations.", "Progressivism"],
    ["Negotiations are key to achieving a two-state solution and lasting peace.", "Moderate"],
    ["The right of return for Palestinian refugees is a fundamental issue.", "Progressivism"],
    ["Israel's military actions in Gaza are justified for self-defense.", "Conservatism"],
    ["Climate change requires immediate global action and cooperation.", "Environmentalism"],
    ["Economic policies should prioritize reducing inequality and poverty.", "Socialism"],
    ["Free speech should be protected, even for controversial opinions.", "Libertarianism"],
    ["Immigration should be restricted to preserve national security and culture.", "Conservatism"],
    ["Healthcare is a basic human right and should be accessible to all.", "Progressivism"],
    ["Government surveillance is necessary to prevent terrorism and crime.", "Authoritarianism"],
    ["Education systems need reform to better prepare students for the future.", "Moderate"],
    ["Environmental regulations should be loosened to promote economic growth.", "Conservatism"],
    ["Support for traditional family values is essential for societal stability.", "Conservatism"],
    ["Gun control laws are necessary to reduce gun violence.", "Progressivism"],
    ["Welfare programs should be expanded to support those in need.", "Socialism"],
    ["International trade agreements are crucial for economic growth.", "Moderate"]
]

csv_file_path = "benchmark_data.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_content)

print(f"CSV file saved at {csv_file_path}")

# Function to evaluate benchmark data from a CSV file
def evaluate_benchmark_from_csv(csv_file_path):
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row['Text']
            expected_ideology = row['Expected Ideology']
            predicted_ideology = ask_ideology(text)
            print(f"Text: {text}")
            print(f"Expected Ideology: {expected_ideology}")
            print(f"Predicted Ideology: {predicted_ideology}")
            print("Match" if expected_ideology == predicted_ideology else "Mismatch")
            print("")

# Evaluate the benchmark using the CSV file
evaluate_benchmark_from_csv(csv_file_path)

if __name__ == "__main__":
    # Example text input for real-time ideology identification
    text = input("Enter a text to identify its ideology: ")
    ideology = ask_ideology(text)
    print(f"\nIdeology of the given text: {ideology}")
