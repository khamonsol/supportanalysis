import logging
import pandas as pd
from models.zero_shot_classifier import ZeroShotEmailClassifier
from data.preprocessing import load_and_clean_data, classify_yes_energy_emails


def main():
    logging.basicConfig(level=logging.INFO)

    # Load and clean data
    file_path = 'support_emails.csv'
    logging.info(f"Loading and cleaning data from {file_path}...")
    data = load_and_clean_data(file_path)

    # Classify YES Energy emails and get remaining data
    logging.info("Classifying YES Energy emails...")
    remaining_data = classify_yes_energy_emails(data)

    # Initialize classifier
    classifier = ZeroShotEmailClassifier()

    # Perform classification based on dataset size
    dataset_size = len(remaining_data)
    emails = remaining_data['Body'].tolist()
    subjects = remaining_data['Subject'].tolist()
    recipients = remaining_data['Cleaned_Recipients'].tolist()

    if dataset_size < 100:
        logging.info("Classifying using naive approach...")
        classifications, confidences = classifier.classify_naive(emails, subjects, recipients)
    elif 100 <= dataset_size < 1000:
        logging.info("Classifying using batch mode...")
        classifications, confidences = classifier.classify_batch_mode(emails, subjects, recipients)
    else:
        logging.info("Classifying using parallel approach...")
        classifications, confidences = classifier.classify_parallel(emails, subjects, recipients)

    # Update the remaining data with classifications and confidences
    remaining_data['Category'] = classifications
    remaining_data['Confidence'] = confidences

    # Combine YES Energy and remaining classified data
    classified_data = pd.concat([data[data['Category'] == 'YES Energy'], remaining_data])

    # Export classified data to a new CSV file
    output_file = 'classified_support_emails.csv'
    logging.info(f"Exporting classified data to {output_file}...")
    classified_data.to_csv(output_file, index=False)

    logging.info("Process completed.")


if __name__ == '__main__':
    main()
