import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split


def train_model(data, labeled_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Train/test split
    train_size = int(0.8 * len(texts))
    val_size = len(texts) - train_size
    train_dataset = torch.utils.data.TensorDataset(inputs['input_ids'][:train_size],
                                                   inputs['attention_mask'][:train_size], labels[:train_size])
    val_dataset = torch.utils.data.TensorDataset(inputs['input_ids'][train_size:],
                                                 inputs['attention_mask'][train_size:], labels[train_size:])

    # Define model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
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
