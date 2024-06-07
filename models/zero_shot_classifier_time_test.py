import unittest
import time
from models.zero_shot_classifier import ZeroShotEmailClassifier

@unittest.skip("Skipping this test- it's intent is to test timings by a dev, not to run during CI")
class TestZeroShotClassificationTimeEstimation(unittest.TestCase):
    def setUp(self):
        # Generate a large dataset for testing
        self.sample_size = 100  # Adjust this value as needed for testing
        self.sample_emails = ["I am having trouble accessing the VPN."] * self.sample_size
        self.sample_subjects = ["VPN Access Issue"] * self.sample_size
        self.sample_recipients = ["it-support@example.com"] * self.sample_size
        self.classifier = ZeroShotEmailClassifier(batch_size=64, max_workers=8)

    def test_zero_shot_classification_naive_time_estimation(self):
        # Measure time for naive classification
        start_time = time.time()
        classifications = self.classifier.classify_naive(
            self.sample_emails, self.sample_subjects, self.sample_recipients
        )
        end_time = time.time()

        # Calculate time taken for the classification
        classification_time = end_time - start_time
        print(
            f"Time taken for naive classification of {len(self.sample_emails)} emails: {classification_time:.4f} seconds")

        # Estimate total time for 2000 emails
        total_emails = 2000
        estimated_total_time = (classification_time / len(self.sample_emails)) * total_emails
        print(f"Estimated total time for {total_emails} emails: {estimated_total_time / 60:.2f} minutes")

        # Assert that the estimated total time is within a reasonable range (e.g., less than 30 minutes)
        self.assertLess(estimated_total_time, 30 * 60)

    def test_zero_shot_classification_batch_time_estimation(self):
        # Measure time for batch classification
        start_time = time.time()
        classifications = self.classifier.classify_batch_mode(
            self.sample_emails, self.sample_subjects, self.sample_recipients
        )
        end_time = time.time()

        # Calculate time taken for the classification
        classification_time = end_time - start_time
        print(
            f"Time taken for batch classification of {len(self.sample_emails)} emails: {classification_time:.4f} seconds")

        # Estimate total time for 2000 emails
        total_emails = 2000
        estimated_total_time = (classification_time / len(self.sample_emails)) * total_emails
        print(f"Estimated total time for {total_emails} emails: {estimated_total_time / 60:.2f} minutes")

        # Assert that the estimated total time is within a reasonable range (e.g., less than 30 minutes)
        self.assertLess(estimated_total_time, 30 * 60)

    def test_zero_shot_classification_parallel_time_estimation(self):
        # Measure time for parallel classification
        start_time = time.time()
        classifications = self.classifier.classify_parallel(
            self.sample_emails, self.sample_subjects, self.sample_recipients
        )
        end_time = time.time()

        # Calculate time taken for the classification
        classification_time = end_time - start_time
        print(
            f"Time taken for parallel classification of {len(self.sample_emails)} emails: {classification_time:.4f} seconds")

        # Estimate total time for 2000 emails
        total_emails = 2000
        estimated_total_time = (classification_time / len(self.sample_emails)) * total_emails
        print(f"Estimated total time for {total_emails} emails: {estimated_total_time / 60:.2f} minutes")

        # Assert that the estimated total time is within a reasonable range (e.g., less than 30 minutes)
        self.assertLess(estimated_total_time, 30 * 60)


if __name__ == '__main__':
    unittest.main()
