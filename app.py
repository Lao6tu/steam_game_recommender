import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
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
st.title("🎮 :rainbow[Steam Game Recommender]")
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
        min_value=1997,
        max_value=2025,
        value=(1997, 2025),
        step=1
    )

    # Multi-select price category
    price_range = st.segmented_control(
        "Price range:",
        options = ['Free', 'Budget', 'Mid-range', 'AAA'],
        selection_mode="multi"
    )

    st.divider()

    # Set up the sidebar radio selection
    selected_model = st.radio(
        "Select model type:",
        options = ["model 1", "model 2", "model 3"],
        captions=[
            "General model",
            "Content based model",
            "Name based model"
        ],
        index=0
    )

# Load data and models
@st.cache_data
def load_data(selected_model):
    try:
        if selected_model == 'model 1': selected_model = 'models_1'
        else: selected_model = 'models_2'

        # Get the absolute path to the directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, selected_model)     
        # Load data from file
        df = pd.read_parquet(os.path.join(models_dir, 'trained_data.parquet'), engine="pyarrow")
        name_to_index = pd.Series(df.index, index=df['name'])
        image_urls = pd.Series(df['header_image'].values, index=df['name']).to_dict()
        
        return df, name_to_index, image_urls
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

# Recommendation function
def get_game_recommendations(game_title, n=10, df=None, name_to_index=None,
                             year_range=(1997,2025), price_range=("Free","AAA"), selected_model='model 1'):
    try: 
        # Year filter
        min_year, max_year = year_range
        filtered_df = df[(df['release_year'] >= min_year) & (df['release_year'] <= max_year)].copy()

        # Price filter (Only filter if at least one price range is selected)
        price_categories = {
            'Free': (0, 0),
            'Budget': (0.01, 10),
            'Mid-range': (10.01, 30),
            'AAA': (30.01, float('inf'))
        }
        if price_range:
            price_conditions = []
            for price_cat in price_range:
                min_p, max_p = price_categories[price_cat]
                price_conditions.append((filtered_df['price'] >= min_p) & (filtered_df['price'] <= max_p))
            filtered_df = filtered_df[np.logical_or.reduce(price_conditions)]
        
        # Cluster Selection    
        idx = name_to_index[game_title]
        cluster_1 = df.iloc[idx]['cluster_1']
        cluster_2 = df.iloc[idx]['cluster_2']
        # Combine cluster assignments
        if selected_model == 'model 1':
            cluster_mask = (filtered_df['cluster_1'] == cluster_1) | (filtered_df['cluster_2'] == cluster_2)
        elif selected_model == 'model 2':
            cluster_mask = (filtered_df['cluster_1'] == cluster_1)
        elif selected_model == 'model 3':
            cluster_mask = (filtered_df['cluster_2'] == cluster_2)
        candidate_indices = filtered_df[cluster_mask].index.tolist()
        if not candidate_indices: candidate_indices = df.index.tolist()
        # Remove the query game itself
        candidate_indices = [i for i in candidate_indices if i != idx]
        # Calculate similarities 
        similarities_1 = cosine_similarity(df.iloc[idx]['latent_1'].reshape(1, -1), 
                                           np.stack(df.loc[candidate_indices, 'latent_1'].values))[0]
        similarities_2 = cosine_similarity(df.iloc[idx]['latent_2'].reshape(1, -1), 
                                           np.stack(df.loc[candidate_indices, 'latent_2'].values))[0]
        # Combine similarities with weights
        if selected_model == 'model 1':
            similarities = 0.4 * similarities_1 + 0.6 * similarities_2
        elif selected_model == 'model 2':
            similarities = similarities_1
        elif selected_model == 'model 3':
            similarities = similarities_2

        # Get top N recommendations
        top_indices = [candidate_indices[i] for i in np.argsort(similarities)[-n:][::-1]]
        recommendations = pd.DataFrame({
            'Game': df.loc[top_indices, 'name'].values,
            'Similarity Score': similarities[np.argsort(similarities)[-n:][::-1]],
            'Price': df.loc[top_indices, 'price'].values,
            'Tags': df.loc[top_indices, 'tags_list'].values,
            'Description': df.loc[top_indices, 'short_description'].values,
            'Release Year': df.loc[top_indices, 'release_year'].values
        })
        return recommendations, None
    except Exception as e:  
        return st.error(f"Error generating recommendations: {str(e)}")

# Load data
df, name_to_index, image_urls = load_data(selected_model)
if df is None or name_to_index is None or image_urls is None:
    st.error("Failed to load required data files. Please check the models directory.")
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
    recommendations, matches = get_game_recommendations(game_query, num_recommendations, df, name_to_index,
                                                        year_range, price_range, selected_model)
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
        st.subheader(f"Games you may like({len(recommendations)} results)")
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
                # Unique key per game
                feedback = st.feedback(
                    "thumbs",
                    key=f"feedback_{row['Game']}",  
                )
            st.divider()
    else:
        st.warning(f"Game '{game_query}' not found in the dataset. Try reduce year range.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("<p class='footer'>Game data updated by March 2025</p>", unsafe_allow_html=True)
st.divider()
