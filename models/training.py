import collections

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

import logging

from models.dataset import CustomDataset


def train_model(data, labeled_data):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Prepare manually labeled data
    texts = []
    labels = []
    label_mapping = {label: idx for idx, label in enumerate(labeled_data.keys())}

    for label, examples in labeled_data.items():
        for example in examples:
            texts.append(example)
            labels.append(label_mapping[label])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    labels = torch.tensor(labels)

    # Check label distribution
    label_counts = collections.Counter(labels.numpy())
    for label, count in label_counts.items():
        logging.info(f"Label {label}: {count} examples")

    # Stratified train/test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, stratify=labels)

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len=512)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len=512)

    # Define model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))
    model.to(device)

    # Training arguments
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train model
    trainer.train()

    return model, tokenizer
