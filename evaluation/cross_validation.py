from sklearn.model_selection import KFold
import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from models.dataset import CustomDataset


def cross_validate_model(data, labeled_data, k=5):
    texts = []
    labels = []
    label_mapping = {label: idx for idx, label in enumerate(labeled_data.keys())}

    for label, examples in labeled_data.items():
        for example in examples:
            texts.append(example)
            labels.append(label_mapping[label])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CustomDataset(texts, labels, tokenizer, max_len=512)

    kf = KFold(n_splits=k)
    results = []

    for train_index, test_index in kf.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, test_index)

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))
        model.to(device)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
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
        results.append(eval_result)

    return np.mean(results, axis=0)

# Example usage of cross-validation function
# results = cross_validate_model(data, manually_labeled_data)
# print(results)
