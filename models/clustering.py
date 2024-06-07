from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def perform_clustering(data, num_clusters=6):
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['Body'])

    # Use KMeans clustering to find common topics
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Assign clusters to emails
    data['Cluster'] = kmeans.labels_

    # Display top terms per cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(num_clusters):
        print(f"Cluster {i}:")
        for ind in order_centroids[i, :10]:
            print(f' {terms[ind]}')

    return data
