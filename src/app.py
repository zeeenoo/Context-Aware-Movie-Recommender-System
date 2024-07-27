import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('../models/camf_model.joblib')

model = load_model()

# Load movie data
@st.cache
def load_movie_data():
    return pd.read_csv('../data/movie.csv')

movies = load_movie_data()

st.title('Context-Aware Movie Recommender System')

user_id = st.number_input('Enter your user ID:', min_value=1, max_value=943, value=1)
hour = st.slider('Select the hour of the day:', 0, 23, 12)
day = st.selectbox('Select the day of the week:', 
                   ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
day_num = day_mapping[day]

if st.button('Get Recommendations'):
    # Create input data for all movies
    input_data = pd.DataFrame({
        'user_id': [user_id] * len(movies),
        'movie_id': movies['movie_id'],
        'hour': [hour] * len(movies),
        'day_of_week': [day_num] * len(movies)
    })

    # Get predictions for all movies
    predictions = model.predict(input_data)

    # Sort movies by predicted rating
    top_movies = movies.iloc[np.argsort(predictions)[::-1][:10]]

    st.subheader('Top 10 Recommended Movies:')
    for i, (_, movie) in enumerate(top_movies.iterrows(), 1):
        st.write(f"{i}. {movie['title']} ({movie['release_date']})")

st.sidebar.info('This recommender system uses context-aware matrix factorization to provide personalized movie recommendations based on user ID, time of day, and day of the week.')