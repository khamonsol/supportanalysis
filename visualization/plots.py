import logging

import matplotlib.pyplot as plt


def plot_email_volume(summary_df):
    # Plot the volume of emails for each category
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Category'], summary_df['Count'], color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Number of Emails')
    plt.title('Volume of Support Emails by Category')
    plt.xticks(rotation=45)
    plt.show()


def plot_automation_impact(summary_df):
    # Plot the potential impact of automation
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Category'], summary_df['Automation Potential'], color='lightgreen')
    plt.xlabel('Category')
    plt.ylabel('Potentially Automatable Emails')
    plt.title('Potential Impact of Automation by Category')
    plt.xticks(rotation=45)
    plt.show()


def plot_classification_results(data):
    # Count the number of emails per category
    category_counts = data['Category'].value_counts()

    # Plot the category distribution
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar')
    plt.title('Email Classification Results')
    plt.xlabel('Category')
    plt.ylabel('Number of Emails')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('classification_results.png')
    plt.show()
    logging.info("Classification results visualization saved as 'classification_results.png'")
