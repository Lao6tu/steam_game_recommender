# steam game recommendation system

## Step 1: Data Preprocessing

The dataset has:
* Text columns (genres, descriptions)
* Numerical columns (peak players, estimated owners, review percentages)

### 1.1 Handling Text Data

* Genres (Multi-label Categorical)
> Steam games often have multiple genres (e.g., "Action, Adventure, RPG").
> Use multi-hot encoding (binary columns for each genre).

* Game Descriptions (NLP)
> Use TF-IDF or Sentence-BERT (SBERT) embeddings to convert descriptions into fixed-length vectors.
> Example: sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings).

### 1.2 Handling Numerical Data

Normalize numerical features (e.g. StandardScaler):

### 1.3 Combine Features

Concatenate:

* Genre multi-hot encodings
* Description embeddings
* Scaled numerical features
