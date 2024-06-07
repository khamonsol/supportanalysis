import unittest
import pandas as pd
from io import StringIO
from data.preprocessing import load_and_clean_data, preprocess_recipients, identify_yes_energy_emails, \
    classify_yes_energy_emails


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample data for preprocess_recipients
        self.recipients = pd.Series([
            'John Doe <john.doe@example.com>, Yes Energy Support <support@yes_energy.com>',
            'Jane Smith <jane.smith@example.com>',
            'support@yesenergy.com',
            'Yes Energy Support <support@yesenergysupport.freshdesk.com>'
        ])
        self.expected_cleaned_recipients = pd.Series([
            'john.doe@example.com, support@yes_energy.com',
            'jane.smith@example.com',
            'support@yesenergy.com',
            'support@yesenergysupport.freshdesk.com'
        ])

        # Sample data for classify_yes_energy_emails
        self.email_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Recipients': ['support@yesenergy.com', 'john.doe@example.com', 'support@yesenergysupport.freshdesk.com'],
            'Subject': ['Subject 1', 'Subject 2', 'Subject 3'],
            'Body': ['Body 1', 'Body 2', 'Body 3']
        })
        self.expected_remaining_data = pd.DataFrame({
            'Date': ['2023-01-02'],
            'Recipients': ['john.doe@example.com'],
            'Subject': ['Subject 2'],
            'Body': ['Body 2'],
            'Cleaned_Recipients': ['john.doe@example.com']
        })

    def test_preprocess_recipients(self):
        cleaned_recipients = preprocess_recipients(self.recipients)
        pd.testing.assert_series_equal(cleaned_recipients, self.expected_cleaned_recipients)

    def test_identify_yes_energy_emails(self):
        cleaned_recipients = preprocess_recipients(self.recipients)
        yes_energy_emails = identify_yes_energy_emails(cleaned_recipients)
        expected_yes_energy_emails = pd.Series([True, False, True, True])
        pd.testing.assert_series_equal(yes_energy_emails, expected_yes_energy_emails)

    def test_classify_yes_energy_emails(self):
        remaining_data = classify_yes_energy_emails(self.email_data)
        expected_yes_energy_data = self.email_data.loc[[0, 2]].copy()
        expected_yes_energy_data['Cleaned_Recipients'] = ['support@yesenergy.com',
                                                          'support@yesenergysupport.freshdesk.com']
        expected_yes_energy_data['Category'] = 'YES Energy'
        expected_yes_energy_data['Confidence'] = 1.0

        # Select only the columns present in the expected data for comparison
        columns_to_compare = ['Date', 'Recipients', 'Subject', 'Body', 'Cleaned_Recipients']
        pd.testing.assert_frame_equal(remaining_data[columns_to_compare].reset_index(drop=True),
                                      self.expected_remaining_data.reset_index(drop=True))
        pd.testing.assert_frame_equal(
            self.email_data.loc[[0, 2], columns_to_compare + ['Category', 'Confidence']].reset_index(drop=True),
            expected_yes_energy_data.reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
