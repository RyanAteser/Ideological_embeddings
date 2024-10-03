
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Load and label debate data based on speakers
def load_debate_data(file_path):
    corpus = []
    labels = []  # Labels for each sentence (0 = Pro-Palestine, 1 = Pro-Israeli, 2 = Neutral)

    with open(file_path, 'r', encoding='utf-8') as file:
        current_speaker = None
        for line in file:
            sentence = line.strip()

            # Detect speaker names and associate with labels
            if sentence.startswith("Benny Morris"):
                current_speaker = 1  # Pro-Israeli
            elif sentence.startswith("Norman Finkelstein"):
                current_speaker = 0  # Pro-Palestine
            elif sentence.startswith("Mouin Rabbani"):
                current_speaker = 0  # Pro-Palestine
            elif sentence.startswith("Steven Bonnell"):
                current_speaker = 1  # Pro-Israeli
            else:
                if current_speaker is not None:
                    corpus.append(sentence)
                    labels.append(current_speaker)

    return corpus, labels

# Load labeled sentences (Pro-Palestinian and Pro-Israeli statements)
def load_predefined_statements(file_path):
    corpus = []
    labels = []  # Labels: (0 = Pro-Palestine, 1 = Pro-Israeli)

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Pro-Palestine"):
                labels.append(0)
                corpus.append(line.replace("Pro-Palestine:", "").strip())
            elif line.startswith("Pro-Israeli"):
                labels.append(1)
                corpus.append(line.replace("Pro-Israeli:", "").strip())

    return corpus, labels

# Load the debate dataset
corpus_debate, labels_debate = load_debate_data('shuffled_transcript_variations_less.txt')

# Load the predefined statements dataset
corpus_predefined, labels_predefined = load_predefined_statements('transcript_variations.txt')

# Combine both datasets
combined_corpus = corpus_debate + corpus_predefined
combined_labels = labels_debate + labels_predefined

# Tokenize the dataset using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Create a Dataset object from the combined data
dataset = Dataset.from_dict({"text": combined_corpus, "label": combined_labels})

# Tokenize the dataset
tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True), batched=True)

# Split into train and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.3, shuffle=True)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,  # Since we only have Pro-Palestine and Pro-Israeli labels
    output_attentions=False,
    output_hidden_states=False
)

# Define class weights for handling imbalances
class_weights = torch.tensor([1.0, 1.0]).to('cuda')  # Adjust weights if needed

# Define a loss function with class weights
def weighted_loss(logits, labels):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    return loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))

# Define training arguments with early stopping and regularization
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    weight_decay=0.01,
    save_total_limit=3,
)

# Metric for evaluation (Accuracy + F1, Precision, and Recall)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.tensor(logits).argmax(-1)
    labels = torch.tensor(labels)

    precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), predictions.cpu(), average='weighted')
    accuracy = (predictions == labels).float().mean().item()

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=lambda data: {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
        'labels': torch.tensor([f['label'] for f in data])
    },
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train the BERT model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./trained_bert_model_hybrid')
tokenizer.save_pretrained('./trained_bert_model_hybrid')

# Convert the trained BERT model to ONNX format
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128), dtype=torch.long).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "bert_model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Model training and ONNX export complete.")
