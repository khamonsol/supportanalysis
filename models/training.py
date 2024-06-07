import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from .dataset import CustomDataset
import logging


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
    dataset = CustomDataset(texts, labels, tokenizer, max_len=512)

    # Train/test split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Define model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,  # Increase the number of epochs
        per_device_train_batch_size=16,  # Adjust batch size
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
