from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification

from .dataset import CustomDataset


def cross_validate_model(data, labeled_data, k=5, use_small_subset=False):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    texts = []
    labels = []
    label_mapping = {label: idx for idx, label in enumerate(labeled_data.keys())}

    for label, examples in labeled_data.items():
        for example in examples:
            texts.append(example)
            labels.append(label_mapping[label])

    if use_small_subset:
        texts = texts[:50]
        labels = labels[:50]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    labels = torch.tensor(labels)

    dataset = CustomDataset(texts, labels, tokenizer, max_len=512)

    kf = KFold(n_splits=k)
    results = []

    for train_index, test_index in kf.split(range(len(dataset))):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, test_index)

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))
        model.to(device)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1 if use_small_subset else 5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()
        eval_result = trainer.evaluate()

        predictions = []
        true_labels = []

        for item in val_dataset:
            text = item['input_ids'].numpy()
            label = item['labels']  # Directly use the label as an integer
            inputs = {'input_ids': torch.tensor([text]).to(device),
                      'attention_mask': torch.tensor([item['attention_mask'].numpy()]).to(device)}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_class_id)
            true_labels.append(label)

        report = classification_report(true_labels, predictions, labels=[0, 1, 2, 3, 4, 5], target_names=[
            'onboarding',
            'beyond web support',
            'missing data collections',
            'general it issues',
            'yes energy',
            'excel uploader support'
        ], zero_division=0)

        print(report)
        results.append(eval_result)

    return np.mean(results, axis=0)

# Example usage of cross-validation function
# results = cross_validate_model(data, manually_labeled_data)
# print(results)
