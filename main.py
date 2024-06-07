from data.preprocessing import load_and_clean_data
from models.clustering import perform_clustering
from models.classification import classify_emails
from data.analysis import analyze_impact
from visualization.plots import plot_email_volume, plot_automation_impact

# Load and preprocess data
data = load_and_clean_data('support_emails.csv')

# Perform clustering to identify common topics
data = perform_clustering(data)

# Classify emails based on clusters
data = classify_emails(data)

# Analyze the potential impact of automation
summary_df = analyze_impact(data)

# Visualize the results
plot_email_volume(summary_df)
plot_automation_impact(summary_df)
