import logging
from concurrent.futures import as_completed, ThreadPoolExecutor
from transformers import pipeline


class ZeroShotEmailClassifier:
    def __init__(self, batch_size=32, max_workers=4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.classifier = self.load_classifier()

    def load_classifier(self):
        logging.info("Loading zero-shot classification pipeline...")
        classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        logging.info("Zero-shot classification pipeline loaded.")
        return classifier

    def get_candidate_labels(self):
        return [
            'Onboarding',
            'Web Support',
            'Missing Data',
            'IT Issues',
            'Excel Support'
        ]

    def classify_emails(self, emails, subjects, recipients):
        results = []
        for email, subject, recipient in zip(emails, subjects, recipients):
            if not email.strip() and not subject.strip() and not recipient.strip():
                logging.warning(f"Skipping empty email: {email}")
                results.append({'labels': ['unknown'], 'scores': [0.0]})
                continue
            combined_text = f"Subject: {subject} Recipients: {recipient} Body: {email}"
            result = self.classifier(combined_text, self.get_candidate_labels())
            results.append(result)
        return results

    def extract_labels_and_confidences(self, results):
        classifications = []
        confidences = []
        for result in results:
            try:
                if 'labels' in result and result['labels']:
                    best_label = result['labels'][0]
                    best_score = result['scores'][0]
                    classifications.append(best_label.split(' (')[0])
                    confidences.append(best_score)
                else:
                    logging.warning(f"Unexpected result format: {result}")
                    classifications.append('unknown')
                    confidences.append(0.0)
            except Exception as e:
                logging.error(f"Error extracting classification: {e}")
                classifications.append('unknown')
                confidences.append(0.0)
        return classifications, confidences

    def classify_naive(self, emails, subjects, recipients):
        results = self.classify_emails(emails, subjects, recipients)
        return self.extract_labels_and_confidences(results)

    def classify_batch_mode(self, emails, subjects, recipients):
        results = []
        for i in range(0, len(emails), self.batch_size):
            batch_emails = emails[i:i + self.batch_size]
            batch_subjects = subjects[i:i + self.batch_size]
            batch_recipients = recipients[i:i + self.batch_size]
            batch_results = self.classify_emails(batch_emails, batch_subjects, batch_recipients)
            results.extend(batch_results)
        return self.extract_labels_and_confidences(results)

    def classify_parallel(self, emails, subjects, recipients):
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.classify_emails, emails[i:i + self.batch_size],
                                subjects[i:i + self.batch_size],
                                recipients[i:i + self.batch_size])
                for i in range(0, len(emails), self.batch_size)
            ]
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    logging.error(f"Error in batch classification: {e}")
        return self.extract_labels_and_confidences(results)
