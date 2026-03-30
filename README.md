# MovieRecommendationSystem 
### Live App : https://movierecommendationsystem-anjal.streamlit.app/


This project builds a content-based movie recommendation system using Natural Language Processing on movie plot descriptions.

The system recommends similar movies based on textual similarity of the `description` field from the Hugging Face dataset `jquigl/imdb-genres`.

## Objective

The goal of this task is to:

- preprocess movie descriptions
- convert text into numerical vectors using TF-IDF or Bag-of-Words
- compute cosine similarity between movies
- recommend the top 5 most similar movies

Pre-trained embeddings are not used.

## Dataset

Source dataset:

- Hugging Face: `jquigl/imdb-genres`

Suggested usage from the task:

- `train` split for building the recommendation model
- `validation` split for testing and simple evaluation

Important note:

- The actual dataset uses `movie title - year` instead of `title`
- The actual dataset uses `rating` instead of `ratings`

The code handles these column differences automatically.


## Approach

### 1. Data Preprocessing

Movie descriptions are cleaned using the following steps:

- convert text to lowercase
- remove punctuation
- remove stopwords

Stemming and lemmatization were kept optional and not applied by default.

### 2. Vectorization

Two methods are supported:

- `TF-IDF` (default)
- `Bag-of-Words`

TF-IDF is used as the default because it gives higher importance to meaningful words and reduces the effect of very common words.

### 3. Recommendation Logic

The recommendation system:

- takes a movie title, vague description, or both
- transforms the query into a vector
- computes cosine similarity with all movies
- optionally filters by genre
- returns the top 5 most similar movies

## Features

- content-based recommendation system
- title-based recommendations
- vague text query support
- optional genre filtering
- TF-IDF and Bag-of-Words support
- notebook version for Google Colab
- Python script version for local execution

## Installation

### Local Setup

1. Clone or download the project.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run Locally with Python

Example using sample data:

```bash
python main.py --source sample --title "Interstellar"
```

Example using a vague description:

```bash
python main.py --source sample --description "space survival mission with astronauts" --genre "Sci-Fi"
```

Example using Hugging Face dataset:

```bash
python main.py --source huggingface --split train --description "space survival mission with astronauts" --genre "Sci-Fi"
```

Example using Bag-of-Words:

```bash
python main.py --source sample --vectorizer bow --description "family animation with music"
```

### Run in Google Colab

1. Open [movie_recommendation_notebook.ipynb](/C:/Users/LANOVO/Downloads/Task/movie_recommendation_notebook.ipynb) in Google Colab.
2. Run the cells one by one.
3. The notebook installs dependencies, loads the dataset, preprocesses text, builds the recommender, and shows example outputs.

## Example Output

For a query like:

```text
space survival mission with astronauts
```

The system can return recommendations like:

```text
Project Kronos - nan
Anti Gravity - nan
Another Plan from Outer Space - 2018
Adiós querida luna - 2004
Calamity - 2016
```

## Evaluation

A simple validation approach is included in the notebook.

The notebook computes an approximate `Precision@5` by checking whether recommended movies share genre tokens with validation examples.

Possible evaluation metrics for future improvement:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- qualitative manual evaluation

## Limitations

- the model only uses the `description` field
- recommendations depend heavily on keyword overlap
- there is no semantic understanding beyond vector similarity
- genre filtering is simple text matching
- no collaborative filtering or user preference data is used

## Future Improvements

- add stemming or lemmatization
- tune TF-IDF parameters
- combine description with metadata such as cast, year, or keywords
- build a small web interface using Streamlit or Gradio
- improve evaluation with stronger relevance labels

## Submission Contents

This submission includes:

- structured Python implementation
- Colab notebook
- dependency file
- sample dataset for offline demo
- README with explanation, setup, and usage



