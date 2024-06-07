import pandas as pd
import re


def load_and_clean_data(file_path):
    # Load CSV data
    data = pd.read_csv(file_path)

    # Normalize and clean text data
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    data['Body'] = data['Body'].apply(clean_text)
    return data
