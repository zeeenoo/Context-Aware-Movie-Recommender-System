from datasets import load_dataset
import pandas as pd

def load_movielens_data():
    """
    Load MovieLens 100K dataset using Hugging Face Datasets.
    
    Returns:
    tuple: ratings DataFrame and movies DataFrame
    """
    dataset = load_dataset("auxten/movielens-20m")
    
    ratings = pd.DataFrame(dataset['train'])
    
    # Extract unique movies and their information
    movies = ratings[['movie_id', 'movie_title']].drop_duplicates().reset_index(drop=True)
    movies.columns = ['movie_id', 'title']
    
    ratings = ratings[['user_id', 'movie_id', 'rating', 'timestamp']]
    
    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies.")
    
    return ratings, movies

if __name__ == "__main__":
    ratings, movies = load_movielens_data()
    
    ratings.to_csv('../data/ratings.csv', index=False)
    movies.to_csv('../data/movies.csv', index=False)
    
    print("Data ingestion complete. Files saved as ratings.csv and movies.csv")