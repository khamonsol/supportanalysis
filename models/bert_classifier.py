import logging

import torch


def classify_emails(data, model, tokenizer):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def predict(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        return predicted_class_id

    category_labels = {
        0: 'Onboarding',
        1: 'Beyond Web Support',
        2: 'Missing Data Collections',
        3: 'General IT Issues',
        4: 'YES Energy',
        5: 'Excel Uploader Support'
    }

    data['Category'] = data['Body'].apply(lambda x: category_labels[predict(x)])
    return data
