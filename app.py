import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

data = pd.merge(ratings, movies, on="movieId")

movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
movie_matrix = movie_matrix.fillna(0)

similarity = cosine_similarity(movie_matrix.T)

similarity_df = pd.DataFrame(similarity,
                             index=movie_matrix.columns,
                             columns=movie_matrix.columns)

def recommend_movies(movie_name):

    similar_movies = similarity_df[movie_name].sort_values(ascending=False)

    return similar_movies.iloc[1:6].index.tolist()

st.title("🎬 Movie Recommendation System")

movie_list = movie_matrix.columns.tolist()

selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):

    recommendations = recommend_movies(selected_movie)

    st.write("### Recommended Movies")

    for movie in recommendations:
        st.write(movie)