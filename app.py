import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process


@st.cache_resource
def load_model():
    with open("models/movie_model.pkl", "rb") as f:
        data = pickle.load(f)

    vectorizer = data["vectorizer"]
    train_df = data["train_df"]

    # Add title_key for fuzzy matching
    train_df['title_key'] = train_df['title'].str.lower().str.strip()

    # Create combined_features safely
    cols_to_combine = [col for col in ['title', 'genres', 'cast', 'description'] if col in train_df.columns]
    train_df['combined_features'] = train_df[cols_to_combine].astype(str).agg(' '.join, axis=1)

    # Transform combined_features to matrix
    train_matrix = vectorizer.transform(train_df['combined_features'])

    return vectorizer, train_matrix, train_df

#loading model once
vectorizer, train_matrix, train_df = load_model()

# recommendation function 
def recommend_movies(title=None, description=None, top_k=5):
    if title:
        title = title.lower().strip()

        match = process.extractOne(
            title,
            train_df['title_key'],
            score_cutoff=60
        )

        if match is None:
            return None, ["No similar movie found"]

        matched_title = match[0]

        idx = train_df[train_df['title_key'] == matched_title].index[0]
        query_vector = train_matrix[idx]

    elif description:
        query_vector = vectorizer.transform([description])
        matched_title = "Based on description"

    else:
        return None, ["Provide input"]

    similarity = cosine_similarity(query_vector, train_matrix).flatten()
    top_indices = similarity.argsort()[::-1][1:top_k+1]

    results = train_df.iloc[top_indices][['title', 'ratings']]

    return matched_title, results

#  Streamlit frontend 
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

option = st.radio("Choose input type:", ["Movie Title", "Description"])

# Title Search 
if option == "Movie Title":
    movie_name = st.text_input("Enter Movie Title")

    if st.button("Recommend"):
        if movie_name:
            matched, results = recommend_movies(title=movie_name)
            if matched is None:
                st.error(results[0])
            else:
                st.success(f"Showing results for: {matched}")
                st.dataframe(results)
        else:
            st.warning("Please enter a movie title")

# Description Search 
else:
    description = st.text_area("Describe a movie you like")
    if st.button("Recommend"):
        if description:
            matched, results = recommend_movies(description=description)
            st.success("Recommendations based on your description")
            st.dataframe(results)
        else:
            st.warning("Please enter description")