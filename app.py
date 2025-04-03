import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set page config first (must be the first Streamlit command)
st.set_page_config(page_title="Game Recommender", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .footer {
        font-size: small;
        color: gray;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Steam Game Recommender")
st.markdown("Discover games similar to your favorites based on our AI-powered recommendation system.")
st.markdown("---")

# Load data and models
@st.cache_data
def load_data():
    try:
        # Get the absolute path to the saved_models directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'saved_models')
        
        # Load all model files from the same directory
        df = pd.read_parquet(os.path.join(models_dir, 'steam_games_features.parquet'), engine="pyarrow")
        cluster_data = np.load(os.path.join(models_dir, "dec_results.npz"))
        df['cluster'] = cluster_data["assignments"]
        latent_data = np.load(os.path.join(models_dir, "latent_features.npz"))
        latent_features = latent_data["features"]
        name_to_index = pd.Series(df.index, index=df['name'])
        
        # Load the games_march2025_cleaned.csv file for image URLs
        image_df = pd.read_csv(os.path.join(current_dir, 'games_march2025_cleaned.csv'), usecols=['name', 'header_image'])
        # Create a dictionary mapping game names to image URLs
        image_urls = pd.Series(image_df['header_image'].values, index=image_df['name']).to_dict()
        
        return df, latent_features, name_to_index, image_urls
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

# Recommendation function
def get_game_recommendations(game_title, n=10, df=None, latent_features=None, name_to_index=None, image_urls=None):
    if df is None or latent_features is None or name_to_index is None:
        return None, ["Data not loaded properly"]
    
    try:
        idx = name_to_index[game_title]
        game_cluster = df.iloc[idx]['cluster']
        game_embed = latent_features[idx].reshape(1, -1)
        
        cluster_indices = df[df['cluster'] == game_cluster].index.tolist()
        cluster_latent = latent_features[cluster_indices]
        similarities = cosine_similarity(game_embed, cluster_latent)[0]
        
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = [cluster_indices[i] for i in similar_indices if cluster_indices[i] != idx][:n]
        
        recommendations = pd.DataFrame({
            'Game': df.iloc[similar_indices]['name'].values,
            'Similarity Score': similarities[np.argsort(similarities)[::-1]][1:n+1],
            'Price': df.iloc[similar_indices]['price'].values,
            'Tags': df.iloc[similar_indices]['tags_list'].values,
            'Description': df.iloc[similar_indices]['short_description'].values
        })
        
        return recommendations, None
    
    except KeyError:
        closest_matches = df[df['name'].str.contains(game_title, case=False, na=False)]['name'].tolist()
        return None, closest_matches[:5]
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None, ["An error occurred while generating recommendations"]

# Load data
df, latent_features, name_to_index, image_urls = load_data()

# Check if data loaded successfully
if df is None or latent_features is None or name_to_index is None or image_urls is None:
    st.error("Failed to load required data files. Please check that all model files exist in the saved_models directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Search Options")
    st.markdown("")
    try:
        game_options = sorted(df['name'].dropna().unique())
        # Set default game to PUBG: BATTLEGROUNDS
        default_game = "PUBG: BATTLEGROUNDS"
        default_index = game_options.index(default_game) if default_game in game_options else 0
        
        game_query = st.selectbox(
            "Search for a game:",
            options=game_options,
            index=default_index,
            help="Start typing to find a game"
        )
    except Exception as e:
        st.error(f"Error loading game list: {str(e)}")
        st.stop()

    st.markdown("")
        
    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10
    )

# Main content
try:
    recommendations, matches = get_game_recommendations(game_query, num_recommendations, df, latent_features, name_to_index, image_urls)

    if recommendations is not None:
        # Show the selected game
        selected_idx = name_to_index[game_query]
        selected_game = df.iloc[selected_idx]
        
        col1, col2 = st.columns([1, 3])
        with col1:
            try:
                # Use the image URL from the dictionary
                if game_query in image_urls and image_urls[game_query]:
                    st.markdown("")
                    st.image(image_urls[game_query], width=200)
                else:
                    st.markdown(f"**{selected_game['name']}**")
                    st.markdown("*Image not available*")
            except:
                st.markdown(f"**{selected_game['name']}**")
                st.markdown("*Image not available*")
        with col2:
            st.subheader(selected_game['name'])
            try:
                price = float(selected_game['price'])
                st.markdown(f"**Price:** ${price:.2f}")
            except:
                st.markdown("**Price:** Not available")
            try:
                tags = eval(selected_game['tags_list']) if isinstance(selected_game['tags_list'], str) else selected_game['tags_list']
                st.markdown(f"**Tags:** {', '.join(tags)}")
            except:
                st.markdown("**Tags:** No tags available")
            try:
                description = selected_game['short_description']
                if pd.isna(description) or description == "":
                    st.markdown("**Description:** No description available")
                else:
                    st.markdown(f"**Description:** {description}")
            except:
                st.markdown("**Description:** No description available")
        
        st.markdown("---")
        st.subheader(f"Recommended Similar Games ({len(recommendations)} results)")
        st.markdown("")
        
        # Display recommendations
        for i, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    # Use the image URL from the dictionary for recommended games
                    game_name = row['Game']
                    if game_name in image_urls and image_urls[game_name]:
                        st.markdown("")
                        st.image(image_urls[game_name], width=200)
                    else:
                        st.markdown(f"*Image not available*")
                except:
                    st.markdown("*Image not available*")
            with col2:
                st.markdown(f"### {row['Game']}")
                st.markdown(f"**Similarity:** {row['Similarity Score']*100:.1f}%")
                try:
                    price = float(row['Price'])
                    st.markdown(f"**Price:** ${price:.2f}")
                except:
                    st.markdown("**Price:** Not available")
                try:
                    tags = eval(row['Tags']) if isinstance(row['Tags'], str) else row['Tags']
                    st.markdown(f"**Tags:** {', '.join(tags)}")
                except:
                    st.markdown("**Tags:** No tags available")
                try:
                    description = row['Description']
                    if pd.isna(description) or description == "":
                        st.markdown("*No description available*")
                    else:
                        st.markdown(f"{description}")
                except:
                    st.markdown("*No description available*")
    else:
        st.warning(f"Game '{game_query}' not found in the dataset.")
        if matches:
            st.info("Did you mean one of these games?")
            for match in matches:
                if st.button(match):
                    st.experimental_rerun()  # Update the query with the selected match
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p class='footer'>Game recommendation system powered by Deep Embedding Clustering</p>", unsafe_allow_html=True)
