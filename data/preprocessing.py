import re
import pandas as pd


def load_and_clean_data(file_path):
    # Load CSV data
    data = pd.read_csv(file_path)

    # Normalize and clean text data
    def clean_text(text):
        if pd.isna(text):
            return ""  # Return empty string for NaN values
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    data['Body'] = data['Body'].apply(clean_text)
    return data


def preprocess_recipients(recipients):
    # Extract email addresses from the recipients string
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    clean_recipients = recipients.apply(lambda x: ', '.join(email_pattern.findall(str(x))))
    return clean_recipients


def identify_yes_energy_emails(cleaned_recipients):
    # Identify emails with domains containing both 'yes' and 'energy'
    yes_energy_pattern = re.compile(r'\b[\w.-]*yes[\w.-]*energy[\w.-]*\b', re.IGNORECASE)
    return cleaned_recipients.apply(lambda x: bool(yes_energy_pattern.search(x)))


def classify_yes_energy_emails(data):
    data['Cleaned_Recipients'] = preprocess_recipients(data['Recipients'])
    yes_energy_emails = identify_yes_energy_emails(data['Cleaned_Recipients'])
    data.loc[yes_energy_emails, 'Category'] = 'YES Energy'
    data.loc[yes_energy_emails, 'Confidence'] = 1.0  # Assume 100% confidence for YES Energy emails
    remaining_data = data[~yes_energy_emails]
    return remaining_data
