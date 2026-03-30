import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model
with open("models/movie_model.pkl", "rb") as f:
    data = pickle.load(f)

vectorizer = data["vectorizer"]
train_matrix = data["train_matrix"]
train_df = data["train_df"]

train_df['title_key'] = train_df['title'].str.lower().str.strip()


def recommend_movies(title=None, description=None, top_k=5):
    if title:
        title = title.lower().strip()
        if title not in train_df['title_key'].values:
            return ["Movie not found!"]

        idx = train_df[train_df['title_key'] == title].index[0]
        query_vector = train_matrix[idx]

    elif description:
        query_vector = vectorizer.transform([description])

    else:
        return ["Please provide a title or description"]

    similarity = cosine_similarity(query_vector, train_matrix).flatten()
    top_indices = similarity.argsort()[::-1][1:top_k+1]

    recommendations = train_df.iloc[top_indices][['title', 'ratings']]
    return recommendations