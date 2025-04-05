# Steam Game Recommendation System (DEC)

The original dataset is from Kaggle [https://www.kaggle.com/datasets/artermiloff/steam-games-dataset](url),\
and powered by Steam Game Scraper Repositories [https://github.com/FronkonGames/Steam-Games-Scraper](url).

This dataset is up to date as of March 2025.\
The shape of original dataset is  (88899, 47), which contains multiple data types such as text, image and numerical data.

## Step 1: Data Preprocessing

The dataset has:
* Text columns (genres, descriptions)
* Numerical columns (peak players, estimated owners, review percentages)

### 1.1 Handling Text Data
* Genres (Multi-label Categorical)
> Steam games often have multiple genres (e.g., "Action, Adventure, RPG").
> Use multi-hot encoding (binary columns for each genre).

* Game Descriptions (NLP)
> Use TF-IDF or Sentence-BERT (SBERT) embeddings to convert descriptions into fixed-length vectors.\
> Example: sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings).

### 1.2 Handling Numerical Data
Normalize numerical features (e.g. StandardScaler):

### 1.3 Combine Features
Concatenate:
* Genre multi-hot encodings
* Description embeddings
* Scaled numerical features

> Example final feature vector:\
`[genre_Action, genre_RPG, ..., desc_embedding_1, ..., peak_players_scaled, ...]`

## Step 2: Dimensionality Reduction (Optional)
If the combined feature dimension is too high (e.g., >1000), apply:

* PCA (for linear reduction)
* UMAP/t-SNE (for non-linear reduction, but slower)

> (DEC will also learn a compressed representation, but pre-reducing helps training stability.)

## Step 3: DEC Training

### 3.1 Autoencoder Pre-training
Train a deep autoencoder to reconstruct the input features.
> Example architecture: 
> Encoder:   Input → Dense(512, ReLU) → Dense(256, ReLU) → Latent (64-dim)  
> Decoder:   Latent → Dense(256, ReLU) → Dense(512, ReLU) → Output (reconstruction)
* Loss: Mean Squared Error (MSE).

### 3.2 Clustering Phase
After pre-training:
1. Remove the decoder, keep the encoder.
2. Initialize cluster centers (using k-means on latent space).
3. Train with KL-divergence loss (DEC loss):
* Compute soft assignments (Q).
* Compute target distribution (P).
* Minimize KL(P || Q).

### 3.3 Hyperparameters
* Number of clusters (k): Use the Elbow Method or Silhouette Score.
* Learning rate: 1e-4 (Adam optimizer).
* Batch size: 256 (for large datasets).

## Step 4: Recommendation System

### 4.1 Assign New Games to Clusters
* Pass a new game’s features through the encoder → get latent embedding.
* Compute Q (soft assignment probabilities) → assign to the most likely cluster.

### 4.2 Recommend Similar Games
For a given game G in cluster C:
* Find the N nearest neighbors (in latent space) within C.
* Return these as recommendations.

### 4.3 Hybrid Approach (Optional)
Combine DEC with collaborative filtering:
* Use DEC clusters as a content-based filter.
* Add user-play history for personalized recommendations.

## Step 5: Evaluation
Since there are no ground-truth labels (unsupervised), use:
* Silhouette Score (higher = better-separated clusters).
* Cluster Coherence (e.g., do games in a cluster share similar genres?).
* User Feedback (A/B testing recommendations).

# Latent Space Data Clusters
![data_structure](https://github.com/user-attachments/assets/0faa5211-4e06-4b89-ac25-29e35b7c6ebe)

