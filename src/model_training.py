import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

class CAMF:
    def __init__(self, n_factors=20, n_epochs=100, lr=0.01, reg=0.01):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
    
    def fit(self, X, y):
        self.n_users = X['userId'].max() + 1
        self.n_items = X['movieId'].max() + 1
        self.n_hours = 24
        self.n_days = 7
        
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.hour_bias = np.zeros(self.n_hours)
        self.day_bias = np.zeros(self.n_days)
        
        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                user = pd.Series(X['userId']).iloc[i]
                item = pd.Series(X['movieId']).iloc[i]
                hour = pd.Series(X['hour']).iloc[i]
                day = pd.Series(X['day_of_week']).iloc[i]
                rating = pd.Series(y).iloc[i]
                
                pred = np.dot(self.user_factors[user], self.item_factors[item]) + self.hour_bias[hour] + self.day_bias[day]
                error = rating - pred
                
                self.user_factors[user] += self.lr * (error * self.item_factors[item] - self.reg * self.user_factors[user])
                self.item_factors[item] += self.lr * (error * self.user_factors[user] - self.reg * self.item_factors[item])
                self.hour_bias[hour] += self.lr * (error - self.reg * self.hour_bias[hour])
                self.day_bias[day] += self.lr * (error - self.reg * self.day_bias[day])
            
            if epoch % 10 == 0:
                train_rmse = np.sqrt(mean_squared_error(y, self.predict(X)))
                print(f"Epoch {epoch}: train RMSE = {train_rmse:.4f}")
    
    def predict(self, X):
        return np.array([
            np.dot(self.user_factors[user], self.item_factors[item]) + self.hour_bias[hour] + self.day_bias[day]
            for user, item, hour, day in zip(X['userId'], X['movieId'], X['hour'], X['day_of_week'])
        ])

def train_model():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load prepared data
    X_train_path = os.path.join(script_dir, '..', 'data', 'X_train.npy')
    y_train_path = os.path.join(script_dir, '..', 'data', 'y_train.npy')
    X_train = np.load(X_train_path, allow_pickle=True)
    y_train = np.load(y_train_path)
    
    # Convert numpy array back to DataFrame
    X_train = pd.DataFrame(X_train, columns=['userId', 'movieId', 'hour', 'day_of_week'])
    
    # Initialize and train the model
    model = CAMF(n_factors=20, n_epochs=100, lr=0.01, reg=0.01)
    model.fit(X_train, y_train)
    
    # Save the trained model
    CamfModel_path = os.path.join(script_dir, '..', 'models', 'camf_model.joblib')
    joblib.dump(model, CamfModel_path)
    print("Model training complete. Model saved as camf_model.joblib")

if __name__ == "__main__":
    train_model()