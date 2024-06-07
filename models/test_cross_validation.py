import unittest
from unittest.mock import patch, MagicMock
from models.cross_validation import cross_validate_model
import torch


class TestCrossValidation(unittest.TestCase):

    @patch('models.cross_validation.BertForSequenceClassification')
    @patch('models.cross_validation.Trainer')
    @patch('models.cross_validation.CustomDataset')
    def test_cross_validation(self, MockDataset, MockTrainer, MockModel):
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 50
        mock_dataset.__getitem__.return_value = {
            'input_ids': torch.tensor([1, 2, 3]),
            'attention_mask': torch.tensor([1, 1, 1]),
            'labels': 0  # Directly use an integer for labels
        }
        MockDataset.return_value = mock_dataset

        # Mock model
        mock_model = MockModel.return_value
        mock_model.to.return_value = None

        def mock_forward(*args, **kwargs):
            return MagicMock(logits=torch.tensor([[0.1, 0.2, 0.7]]))

        mock_model.forward.side_effect = mock_forward

        # Mock trainer
        mock_trainer = MockTrainer.return_value
        mock_trainer.evaluate.return_value = {'eval_loss': 0.5}
        mock_trainer.train.return_value = None

        # Prepare test data
        test_data = []
        manually_labeled_data = {
            'onboarding': ['signing nda', 'iso submission setup', 'new database user setup'] * 10,
            'beyond web support': ['trouble loading web based reports in beyond', 'access to reports'] * 10,
            'missing data collections': ['missing awards submittals', 'missing invoices'] * 10,
            'general it issues': ['vpn access', 'shared drive access', 'general computer problems'] * 10,
            'yes energy': ['yes energy issues', 'contacting yes energy support', 'issues with yes energy data'] * 10,
            'excel uploader support': ['bespoke excel spreadsheets', 'scripts to upload to market'] * 10
        }

        # Perform cross-validation with a small subset
        result = cross_validate_model(test_data, manually_labeled_data, use_small_subset=True)

        # Check that evaluate was called
        mock_trainer.evaluate.assert_called()
        self.assertIn('eval_loss', result)


if __name__ == '__main__':
    unittest.main()
