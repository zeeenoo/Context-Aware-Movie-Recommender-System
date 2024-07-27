import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(ratings_file, movies_file):
    """
    Prepare data for the context-aware recommender system.
    
    Args:
    ratings_file (str): Path to the ratings CSV file.
    movies_file (str): Path to the movies CSV file.
    
    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    # Load data
    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)
    
    # Convert timestamp to datetime and extract contextual features
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'])
    ratings['hour'] = ratings['timestamp'].dt.hour
    ratings['day_of_week'] = ratings['timestamp'].dt.dayofweek
    
    # Merge ratings with movie information
    data = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
    
    # Prepare features and target
    X = data[['userId', 'movieId', 'hour', 'day_of_week']]
    y = data['rating']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    ratings_file = '../data/rating.csv'
    movies_file = '../data/movie.csv'
    
    X_train, X_test, y_train, y_test = prepare_data(ratings_file, movies_file)
    
    # Save prepared data
    np.save('../data/X_train.npy', X_train)
    np.save('../data/X_test.npy', X_test)
    np.save('../data/y_train.npy', y_train)
    np.save('../data/y_test.npy', y_test)
    
    print("Data preparation complete. Prepared data saved as numpy arrays.")