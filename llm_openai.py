import openai

# Set your OpenAI API key here
openai.api_key = 'your-api-key-here'

# Function to classify the ideology of a statement
def classify_ideology(statement):
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"Classify the following statement as 'Pro-Palestinian,' 'Pro-Israeli,' or 'Neutral': {statement}",
        max_tokens=10
    )
    return response.choices[0].text.strip()

# Function to analyze the transcript
def analyze_transcript(transcript_file):
    with open(transcript_file, 'r') as file:
        lines = file.readlines()

    current_speaker = None
    speaker_statements = {}

    for line in lines:
        # Check if the line is a speaker label
        if line.startswith("Speaker"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                current_speaker = parts[0].strip()
                statement = parts[1].strip()

                if current_speaker not in speaker_statements:
                    speaker_statements[current_speaker] = []

                speaker_statements[current_speaker].append(statement)
        elif current_speaker:
            # Continue the statement if it spills onto the next line
            speaker_statements[current_speaker][-1] += " " + line.strip()

    # Analyze the statements by speaker
    for speaker, statements in speaker_statements.items():
        print(f"{speaker}'s ideology:")
        for statement in statements:
            result = classify_ideology(statement)
            print(f"Statement: {statement}\nClassification: {result}\n")

# Analyze the uploaded transcript
transcript_file_path = 'transcript.txt'
analyze_transcript(transcript_file_path)
