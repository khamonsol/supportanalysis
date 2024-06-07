def classify_emails(data):
    # Define categories based on clusters
    cluster_labels = {
        0: 'Onboarding',
        1: 'Beyond Web Support',
        2: 'Missing Data Collections',
        3: 'General IT Issues',
        4: 'YES Energy',
        5: 'Excel Uploader Support'
    }

    # Map cluster labels to categories
    data['Category'] = data['Cluster'].map(cluster_labels)
    return data
