import torch
import numpy as np
from transformers import BertTokenizer, pipeline
import onnxruntime as ort
import torch.nn.functional as F
from collections import defaultdict
import re

# Load trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./trained_bert_model_hybrid')

# Load ONNX model for inference
onnx_model_path = 'bert_model.onnx'
session = ort.InferenceSession(onnx_model_path)

# Define function to run ONNX inference and extract both prediction and confidence
def predict_with_onnx(input_data):
    inputs = {'input': input_data}
    outputs = session.run(None, inputs)

    # Assuming the outputs contain logits, apply softmax to get confidence
    logits = outputs[0]  # First output from the model (logits)
    probs = F.softmax(torch.tensor(logits), dim=-1)  # Apply softmax to get probabilities
    confidence, prediction = torch.max(probs, dim=-1)  # Get max confidence and corresponding class

    return prediction.item(), confidence.item()

# Load GPT-Neo model for fallback
llm = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0, max_new_tokens=128)

# GPT-Neo based fallback function with stricter guidelines for answer
def gpt_neo_fallback(sentences):
    prompt = f"Is the following statement Pro-Israeli, Pro-Palestinian, or Neutral? Please respond with just one word: 'Pro-Israeli', 'Pro-Palestinian', or 'Neutral'. Sentence: {sentences}."
    result = llm(prompt)[0]['generated_text']
    # Ensure only one-word answer is extracted
    if 'Pro-Israeli' in result:
        return 'Pro-Israeli'
    elif 'Pro-Palestinian' in result:
        return 'Pro-Palestinian'
    elif 'Neutral' in result:
        return 'Neutral'
    else:
        return 'Neutral'  # Default to neutral if unclear

# Hybrid prediction function with expanded context window
def hybrid_predict_contextual_with_history(previous_sentences, current_sentence, llm_confidence_threshold=0.6):
    # Combine previous sentences and the current sentence for more context
    full_input = ' '.join(previous_sentences[-10:] + [current_sentence])  # Limit history to the last 10 sentences

    # Tokenize input
    input_data = tokenizer(full_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)['input_ids'].numpy()

    # Predict using ONNX model
    prediction, confidence = predict_with_onnx(input_data)

    # Log confidence level for debugging
    print(f"ONNX model confidence: {confidence}")

    # Trigger GPT-Neo if confidence is low or borderline
    if confidence < llm_confidence_threshold or (0.5 <= confidence <= 0.6):
        print("Triggering GPT-Neo due to low/medium confidence.")
        return gpt_neo_fallback(full_input)
    else:
        return "Pro-Israeli" if prediction == 1 else "Pro-Palestinian" if prediction == 0 else "Neutral"

# Process a new transcript and aggregate results per speaker
def process_transcript_with_context(transcript_file, hybrid_predict_contextual_with_history):
    speaker_stats = defaultdict(lambda: {'Pro-Palestine': 0, 'Pro-Israeli': 0, 'Neutral': 0, 'total_sentences': 0})
    speaker_ideologies = {}

    current_speaker = None
    sentences = []
    history_sentences = []

    with open(transcript_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # Regular expression to match the speaker format with timestamps
            match = re.match(r"^(Speaker \d+) \[\d+.\d+s - \d+.\d+s\]:\s+(.*)", line)
            if match:
                # If we have collected sentences for the previous speaker, process them
                if sentences and current_speaker:
                    label = hybrid_predict_contextual_with_history(history_sentences, ' '.join(sentences))

                    # Ensure the label is valid
                    if label in speaker_stats[current_speaker]:
                        speaker_stats[current_speaker][label] += len(sentences)  # Count all sentences as the predicted label
                        speaker_stats[current_speaker]['total_sentences'] += len(sentences)

                    # Add sentences to the history for future predictions
                    history_sentences.extend(sentences)
                    sentences = []  # Reset sentence list for the next speaker

                # Identify the new speaker
                current_speaker, sentence = match.groups()
                sentences.append(sentence)  # Collect the first sentence from the new speaker
            else:
                # Continuation of the same speaker's text
                if line:
                    sentences.append(line)

    # Process any remaining sentences after the loop ends
    if sentences and current_speaker:
        label = hybrid_predict_contextual_with_history(history_sentences, ' '.join(sentences))

        # Ensure the label is valid
        if label in speaker_stats[current_speaker]:
            speaker_stats[current_speaker][label] += len(sentences)
            speaker_stats[current_speaker]['total_sentences'] += len(sentences)

    # For each speaker, predict the overall ideology based on the percentage of each label
    for speaker, stats in speaker_stats.items():
        pro_palestine_pct = (stats['Pro-Palestine'] / stats['total_sentences']) * 100 if stats['total_sentences'] > 0 else 0
        pro_israeli_pct = (stats['Pro-Israeli'] / stats['total_sentences']) * 100 if stats['total_sentences'] > 0 else 0

        if pro_palestine_pct > 60:
            speaker_ideologies[speaker] = "Pro-Palestine"
        elif pro_israeli_pct > 60:
            speaker_ideologies[speaker] = "Pro-Israeli"
        else:
            speaker_ideologies[speaker] = "Neutral/Mixed"

    return speaker_stats, speaker_ideologies

# Function to print the cumulative summary
def print_cumulative_summary(speaker_stats, speaker_ideologies):
    print("Cumulative Classification and Ideology Prediction per Speaker:")
    for speaker, stats in speaker_stats.items():
        pro_palestine_pct = (stats['Pro-Palestine'] / stats['total_sentences']) * 100 if stats['total_sentences'] > 0 else 0
        pro_israeli_pct = (stats['Pro-Israeli'] / stats['total_sentences']) * 100 if stats['total_sentences'] > 0 else 0
        neutral_pct = (stats['Neutral'] / stats['total_sentences']) * 100 if stats['total_sentences'] > 0 else 0

        print(f"Speaker: {speaker}")
        print(f"Pro-Palestine: {pro_palestine_pct:.2f}% ({stats['Pro-Palestine']} sentences)")
        print(f"Pro-Israeli: {pro_israeli_pct:.2f}% ({stats['Pro-Israeli']} sentences)")
        print(f"Neutral: {neutral_pct:.2f}% ({stats['Neutral']} sentences)")
        print(f"Predicted Ideology: {speaker_ideologies[speaker]}")
        print("-" * 50)

# Example usage
transcript_file = 'transcript.txt'  # Adjust the file name as needed

speaker_stats, speaker_ideologies = process_transcript_with_context(transcript_file, hybrid_predict_contextual_with_history)

# Print the cumulative summary of the predictions
print_cumulative_summary(speaker_stats, speaker_ideologies)

# Save results to file
with open('prediction_results.txt', 'w') as f:
    for speaker, stats in speaker_stats.items():
        f.write(f"Speaker: {speaker}\n")
        f.write(f"Pro-Palestine: {stats['Pro-Palestine']} sentences\n")
        f.write(f"Pro-Israeli: {stats['Pro-Israeli']} sentences\n")
        f.write(f"Neutral: {stats['Neutral']} sentences\n")
        f.write(f"Predicted Ideology: {speaker_ideologies[speaker]}\n")
        f.write("-" * 50 + "\n")

print("Predictions saved to 'prediction_results.txt'.")
