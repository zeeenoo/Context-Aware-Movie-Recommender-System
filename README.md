# Context-Aware Movie Recommender System
This project implements a context-aware movie recommender system using the MovieLens 20M dataset. It uses Context-Aware Matrix Factorization (CAMF) to provide personalized movie recommendations based on user ID, time of day, and day of the week.

## Project Structure

```
context_aware_recommender/
│
├── docs/
│   ├── ProjectCharter.md
│   └── DeploymentInstructions.md
│
├── notebooks/
│   ├── DataExploration.ipynb
│   └── EvaluationMetrics.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── app.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (for deployment)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/zeeenoo/Context-Aware-Movie-Recommender-System.git
   cd context-aware-recommender
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Data Preparation:
   ```
   python src/data_preparation.py
   ```

2. Model Training:
   ```
   python src/model_training.py
   ```

3. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```

## Deployment

To deploy the application using Docker, follow the instructions in `docs/DeploymentInstructions.md`.

## Dataset

This project uses the the MovieLens 20M dataset, which is loaded directly from Kaggle Datasets. The dataset includes movies rating, user information, and timestamps, allowing for context-aware recommendations.

## Acknowledgments

- the MovieLens 20M dataset provided by kaggle Datasets
- Inspired by research on context-aware recommender systems
- Streamlit for providing an easy-to-use framework for building data applications

For more information on the project goals, timeline, and success criteria, please refer to the `ProjectCharter.md` file in the `docs` directory.