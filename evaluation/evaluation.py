import torch
from sklearn.metrics import classification_report


def evaluate_model(data, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = data['Body'].tolist()
    true_labels = data['Category'].map({
        'Onboarding': 0,
        'Beyond Web Support': 1,
        'Missing Data Collections': 2,
        'General IT Issues': 3,
        'YES Energy': 4,
        'Excel Uploader Support': 5
    }).tolist()

    predictions = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predictions.append(predicted_class_id)

    report = classification_report(true_labels, predictions, target_names=[
        'Onboarding',
        'Beyond Web Support',
        'Missing Data Collections',
        'General IT Issues',
        'YES Energy',
        'Excel Uploader Support'
    ])
    print(report)
