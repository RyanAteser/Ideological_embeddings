import os
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, ConfusionMatrixDisplay, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to query GPT-2 for ideology
def query_gpt2(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    input_text = f"Identify the ideology of the following sentence in one word (Pro-Israeli, Pro-Palestine, Neutral, etc.): '{text}'"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=10,
        num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id,
        temperature=1.0,
        top_p=0.85,
        do_sample=True
    )

    output_texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    print(f"Generated Outputs: {output_texts}")

    possible_ideologies = ["Pro-Israeli", "Pro-Palestine", "Neutral", "Neutral, leans Pro-Israeli", "Neutral, leans Pro-Palestine"]

    for output_text in output_texts:
        for ideology in possible_ideologies:
            if ideology in output_text:
                return ideology

    return output_texts[0]

# Function to read the corpus and labels from a text file with labels in parentheses
def load_corpus_and_labels(file_path):
    corpus = []
    labels = []
    label_map = {
        "Pro-Israeli": 0,
        "Pro-Palestine": 1
    }

    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r"^(.*?):\s+(.*)$", line.strip())
            if match:
                label, text = match.groups()
                if label in label_map:
                    labels.append(label_map[label])
                    corpus.append(text)

    return corpus, labels

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
        inputs = {key: val.squeeze(0).to(device) for key, val in inputs.items()}  # Move inputs to GPU
        inputs['labels'] = torch.tensor(label, dtype=torch.long).to(device)
        return inputs

# Function to identify the ideology from a given text using your fine-tuned BERT
def identify_ideology_bert(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    model.to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    alignment_prob = torch.nn.functional.softmax(logits, dim=1).max().item()

    ideologies = ["Pro-Israeli", "Pro-Palestine"]
    return logits, predicted_class_id, ideologies[predicted_class_id], alignment_prob

# Function to compare with baseline BERT model and validate using GPT-2
def compare_with_baseline(text, model, tokenizer):
    _, _, my_model_ideology, _ = identify_ideology_bert(text, model, tokenizer)
    gpt2_ideology = query_gpt2(text)

    print(f"My Model Ideology: {my_model_ideology}")
    print(f"GPT-2 Ideology: {gpt2_ideology}")

    if my_model_ideology != gpt2_ideology:
        print("Discrepancy found between GPT-2 and my model. Using GPT-2 prediction.")
        return gpt2_ideology
    return my_model_ideology

# Function to evaluate the model performance
def evaluate_model(eval_texts, eval_labels, model, tokenizer):
    model.eval()
    predictions = []
    true_labels = []
    for text, label in zip(eval_texts, eval_labels):
        _, predicted_class_id, _, _ = identify_ideology_bert(text, model, tokenizer)
        predictions.append(predicted_class_id)
        true_labels.append(label)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}")

    conf_matrix = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Pro-Israeli", "Pro-Palestine"])
    disp.plot(cmap='Blues')
    plt.show()
def is_novel(position, archive, threshold=0.5):
    position_array = np.array(position, dtype=float)
    for archived_array in archive:
        archived_array = np.array(archived_array, dtype=float)
        if np.linalg.norm(position_array - archived_array) < threshold:
            return False
    return True

# Dynamic Learning with Exploration and GPT-2 Feedback
def dynamic_learning_with_gpt2(corpus, labels, model, tokenizer, epochs=5, exploration_threshold=0.50):
    archive = defaultdict(list)
    for epoch in range(epochs):
        print(f"Dynamic Learning Epoch {epoch + 1}/{epochs}")
        for text, label in zip(corpus, labels):
            logits, predicted_class_id, ideology_label, alignment_prob = identify_ideology_bert(text, model, tokenizer)

            if is_novel(logits.detach().cpu().numpy(), archive[ideology_label], exploration_threshold):
                archive[ideology_label].append(logits.detach().cpu().numpy())

            final_ideology = compare_with_baseline(text, model, tokenizer)

            if final_ideology != ideology_label:
                print(f"Updating model based on feedback for: {text}")
                corpus.append(text)
                labels.append(label)

                train_texts, eval_texts, train_labels, eval_labels = train_test_split(corpus, labels, test_size=0.2, random_state=42)
                train_dataset = IdeologyDataset(train_texts, train_labels, tokenizer)
                trainer.train_dataset = train_dataset
                trainer.train()

    print("Dynamic learning with GPT-2 feedback completed.")

if __name__ == "__main__":
    # Load the corpus and labels from the text file
    corpus, labels = load_corpus_and_labels('transcript_variations.txt')

    print(f"Loaded {len(corpus)} samples.")

    # Load pre-trained BERT model and tokenizer from Hugging Face
    print("Loading BERT model and tokenizer...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Split data into training and evaluation sets
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(corpus, labels, test_size=0.2, random_state=42)

    train_dataset = IdeologyDataset(train_texts, train_labels, tokenizer)
    eval_dataset = IdeologyDataset(eval_texts, eval_labels, tokenizer)

    # Fine-tune the BERT model
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        fp16=True,  # Enable mixed precision training
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

    # Evaluate the model
    evaluate_model(eval_texts, eval_labels, model, tokenizer)

    # Execute dynamic learning with GPT-2 feedback
    dynamic_learning_with_gpt2(corpus, labels, model, tokenizer)

