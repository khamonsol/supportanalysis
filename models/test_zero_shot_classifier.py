import unittest
from typing import List
from dataclasses import dataclass
from models.zero_shot_classifier import ZeroShotEmailClassifier


@dataclass
class ClassifierTestCase:
    body: List[str]
    subject: List[str]
    recipients: List[str]
    expected_classification: str


class TestZeroShotEmailClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = ZeroShotEmailClassifier()

        simple = ClassifierTestCase(
            body=["I am having trouble accessing the VPN."],
            subject=["VPN Access Issue"],
            recipients=["it-support@example.com"],
            expected_classification="IT Issues"
        )
        onboarding = ClassifierTestCase(
            body=["The new hire needs access to the database."],
            subject=["Database Access for New Hire"],
            recipients=["onboarding@example.com"],
            expected_classification="Onboarding"
        )

        self.cases = [simple, onboarding]

    def test_classify_naive(self):
        for case in self.cases:
            with self.subTest(case=case):
                classifications, confidences = self.classifier.classify_naive(
                    case.body, case.subject, case.recipients)
                self.assertIsInstance(classifications, list)
                self.assertIsInstance(confidences, list)
                self.assertEqual(len(classifications), len(case.body))
                self.assertEqual(len(confidences), len(case.body))

                # Check specific classification and confidence
                self.assertEqual(classifications[0], case.expected_classification)
                self.assertGreater(confidences[0], 0)

    def test_classify_batch_mode(self):
        for case in self.cases:
            with self.subTest(case=case):
                classifications, confidences = self.classifier.classify_batch_mode(
                    case.body, case.subject, case.recipients)
                self.assertIsInstance(classifications, list)
                self.assertIsInstance(confidences, list)
                self.assertEqual(len(classifications), len(case.body))
                self.assertEqual(len(confidences), len(case.body))

                # Check specific classification and confidence
                self.assertEqual(classifications[0], case.expected_classification)
                self.assertGreater(confidences[0], 0)

    def test_classify_parallel(self):
        for case in self.cases:
            with self.subTest(case=case):
                classifications, confidences = self.classifier.classify_parallel(
                    case.body, case.subject, case.recipients)
                self.assertIsInstance(classifications, list)
                self.assertIsInstance(confidences, list)
                self.assertEqual(len(classifications), len(case.body))
                self.assertEqual(len(confidences), len(case.body))

                # Check specific classification and confidence
                self.assertEqual(classifications[0], case.expected_classification)
                self.assertGreater(confidences[0], 0)


if __name__ == '__main__':
    unittest.main()
