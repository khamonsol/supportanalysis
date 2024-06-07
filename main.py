from data.preprocessing import load_and_clean_data
from models.training import train_model
from models.bert_classifier import classify_emails
from data.analysis import analyze_impact
from visualization.plots import plot_email_volume, plot_automation_impact

# Load and preprocess data
data = load_and_clean_data('support_emails.csv')

# Manually labeled data for initial training
manually_labeled_data = {
    'Onboarding': [
        'signing NDA',
        'ISO submission setup',
        'new database user setup'
    ],
    'Beyond Web Support': [
        'trouble loading web based reports in Beyond',
        'access to reports'
    ],
    'Missing Data Collections': [
        'missing awards submittals',
        'missing invoices'
    ],
    'General IT Issues': [
        'VPN access',
        'shared drive access',
        'general computer problems'
    ],
    'YES Energy': [
        'YES energy issues'
        'DPA upload to market '
    ],
    'Excel Uploader Support': [
        'excel spreadsheets',
        'upload to market'
    ]
}

# Train the BERT model
model, tokenizer = train_model(data, manually_labeled_data)

# Classify emails using the trained BERT model
data = classify_emails(data, model, tokenizer)

# Analyze the potential impact of automation
summary_df = analyze_impact(data)

# Save the classified emails to a new CSV file
data.to_csv('classified_support_emails.csv', index=False)

# Visualize the results
plot_email_volume(summary_df)
plot_automation_impact(summary_df)
