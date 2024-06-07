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
