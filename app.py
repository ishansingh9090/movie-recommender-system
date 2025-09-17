import streamlit as st
import pandas as pd
import pickle
import requests
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to fetch poster
def fetch_posters(movie_id, retries=3, timeout=10):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=19f988545cda65e8479aaa4d986ad979&language=en-US"

    for i in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            if 'poster_path' in data and data['poster_path']:
                return f"https://image.tmdb.org/t/p/w500/{data['poster_path']}"
            else:
                return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"

        except requests.exceptions.RequestException as e:
            print(f"Attempt {i + 1} failed: {e}")
            time.sleep(2)

    return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"


# Recommendation function
def recommend(movie, movies, similarity):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_posters(movie_id))

    return recommended_movies, recommended_posters


# 🔥 Cache the similarity computation
@st.cache_resource
def load_data_and_similarity():
    movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movie_dict)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity


# Load data + similarity (cached)
movies, similarity = load_data_and_similarity()

# Streamlit UI
st.title("🎬 Movie Recommender System")

selected_movie_name = st.selectbox(
    "Select a movie to get recommendations:",
    movies['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name, movies, similarity)
    st.subheader("Here are your recommendations:")

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            col.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="{posters[idx]}" style="width:150px; height:225px; border-radius:10px;"/>
                    <p style="font-size:14px; font-weight:bold; margin-top:5px; word-wrap:break-word;">
                        {names[idx]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
