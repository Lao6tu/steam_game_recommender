import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Set page config first (must be the first Streamlit command)
st.set_page_config(page_title="Steam Game Recommender", page_icon="🕹️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .footer {
        color: gray;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🕹️ :rainbow[Steam Game Recommender]")
st.markdown("Discover games similar to your favorites based on our AI-powered recommendation system.")
st.markdown("")

# Sidebar
with st.sidebar:  
    st.title("⚙️ :red[Search Options]")
    st.markdown("")

    # Number of Recommendations
    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10
    )
    # Release year filter
    year_range = st.slider(
        "Release year range:",
        min_value=1998,
        max_value=2025,
        value=(1998, 2025),
        step=1
    )
    # Model type selection
    selected_models = st.radio(
        "Select model type:",
        ["models_1", "models_2"],
        captions=[
            "Content based model",
            "Name based model",
        ],
        index=0
    )

# Load data and models
@st.cache_data
def load_data(selected_models):
    try:
        # Get the absolute path to the directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Create the full path to the selected models directory
        models_dir = os.path.join(current_dir, selected_models)
        
        # Load all model files from the same directory
        df = pd.read_parquet(os.path.join(current_dir, 'steam_game_dataset_filtered.parquet'), engine="pyarrow")
        cluster_data = np.load(os.path.join(models_dir, "dec_results.npz"))
        df['cluster'] = cluster_data["assignments"]
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        latent_data = np.load(os.path.join(models_dir, "latent_features.npz"))
        latent_features = latent_data["assignments"]
        name_to_index = pd.Series(df.index, index=df['name'])
        image_urls = pd.Series(df['header_image'].values, index=df['name']).to_dict()
        
        return df, latent_features, name_to_index, image_urls
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

# Recommendation function
def get_game_recommendations(game_title, n=10, df=None, latent_features=None, name_to_index=None, image_urls=None, year_range=(None)):
    if df is None or latent_features is None or name_to_index is None:
        return None, ["Data not loaded properly"]
    
    try:
        idx = name_to_index[game_title]
        game_cluster = df.iloc[idx]['cluster']
        game_embed = latent_features[idx].reshape(1, -1)
        
        min_year, max_year = year_range
        df = df[(df['release_year'] >= min_year) & (df['release_year'] <= max_year)]

        cluster_indices = filtered_df[filtered_df['cluster'] == game_cluster].index.tolist()
        if not cluster_indices:
            st.warning("No games found in the selected year range. Showing recommendations from all years.")
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
df, latent_features, name_to_index, image_urls = load_data(selected_models)

# Check if data loaded successfully
if df is None or latent_features is None or name_to_index is None or image_urls is None:
    st.error("Failed to load required data files. Please check that all model files exist in the saved_models directory.")
    st.stop()

# Select Box
try:
    game_options = df.sort_values('estimated_owners', ascending=False)['name']
    game_query = st.selectbox(
        "**Search Box**",
        options=game_options,
        index=None,
        placeholder="🔍 Search for a steam game...",
        label_visibility='collapsed'
    )
    if not game_query:
        game_query = "Black Myth: Wukong"
except Exception as e:
    st.error(f"Error loading game list: {str(e)}")
    st.stop()

st.divider()

# Main content
try:
    recommendations, matches = get_game_recommendations(game_query, num_recommendations, df, latent_features, name_to_index, image_urls, year_range)

    if recommendations is not None:
        selected_idx = name_to_index[game_query]
        selected_game = df.iloc[selected_idx]
        
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            try:
                if game_query in image_urls and image_urls[game_query]:
                    st.markdown("")
                    st.image(image_urls[game_query])
            except:
                st.markdown("*Image not available*")
        with col2:
            st.subheader(f"{selected_game['name']} :green-badge[{df['release_year'].iloc[selected_idx]}]")
            try:
                price = float(selected_game['price'])
                st.markdown(f"**Price:** ${price:.2f}")
            except:
                st.markdown("**Price: Not available**")
            try:
                tags = eval(selected_game['tags_list']) if isinstance(selected_game['tags_list'], str) else selected_game['tags_list']
                st.markdown(f"**Tags:** {', '.join(tags)}")
            except:
                st.markdown("**Tags:** No tags available")
            try:
                description = selected_game['short_description']
                if pd.isna(description) or description == "":
                    st.markdown("*No description available*")
                else:
                    st.markdown(f"*{description}*")
            except:
                st.markdown("*No description available*")
        
        st.markdown("---")
        st.subheader(f"Recommended Similar Games ({len(recommendations)} results)")
        st.markdown("")
        
        # Display recommendations
        for i, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 2], gap="large")
            with col1:
                try:
                    game_name = row['Game']
                    if game_name in image_urls and image_urls[game_name]:
                        st.markdown("")
                        st.image(image_urls[game_name])
                    else:
                        st.markdown(f"*Image not available*")
                except:
                    st.markdown("*Image not available*")
            with col2:
                st.subheader(f"{row['Game']} :green-badge[{df['release_year'].iloc[name_to_index[game_name]]}]")
                st.markdown(f":red-badge[**Similarity**: {row['Similarity Score']*100:.1f}%]")
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
                        st.markdown(f"*{description}*")
                except:
                    st.markdown("*No description available*")
            st.divider()
    else:
        st.warning(f"Game '{game_query}' not found in the dataset.")
        if matches:
            st.info("Did you mean one of these games?")
            for match in matches:
                if st.button(match):
                    st.experimental_rerun()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("<p class='footer'>Powered by Sentence-Transformer and Deep Embedding Clustering</p>", unsafe_allow_html=True)
st.divider()
