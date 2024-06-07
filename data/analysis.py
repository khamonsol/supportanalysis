import pandas as pd


def analyze_impact(data):
    # Calculate the volume of emails for each category
    category_counts = data['Category'].value_counts()

    # Estimate the potential impact of automation
    automatable_categories = ['onboarding', 'beyond web support', 'missing data collections', 'general it issues',
                              'excel uploader support']
    automation_potential = {category: count * 0.3 for category, count in category_counts.items() if
                            category in automatable_categories}

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Category': category_counts.index,
        'Count': category_counts.values,
        'Automatable': category_counts.index.isin(automatable_categories),
        'Automation Potential': [automation_potential.get(category, 0) for category in category_counts.index]
    })

    return summary_df
