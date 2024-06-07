from sklearn.metrics import classification_report
import torch


def evaluate_model(data, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = data['Body'].tolist()
    true_labels = data['Category'].map({
        'onboarding': 0,
        'beyond web support': 1,
        'missing data collections': 2,
        'general it issues': 3,
        'yes energy': 4,
        'excel uploader support': 5
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

    report = classification_report(true_labels, predictions, labels=[0, 1, 2, 3, 4, 5], target_names=[
        'onboarding',
        'beyond web support',
        'missing data collections',
        'general it issues',
        'yes energy',
        'excel uploader support'
    ], zero_division=0)
    print(report)
