import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

# -----------------------------
# Load Precomputed Data
# -----------------------------
@st.cache_data
def load_data():
    # Load the ratings matrix
    R = pd.read_csv("data/Rmat.csv", index_col=0)

    # Load movie metadata
    movies = pd.read_csv(
        "data/movies.dat",
        sep="::",
        engine='python',
        names=["MovieID", "Title", "Genres"],
        encoding='latin-1'
        )
    movies["MovieID"] = "m" + movies["MovieID"].astype(str)
    movies.set_index("MovieID", inplace=True)

    # Load popularity ranking as a DataFrame
    popular_rank = pd.read_csv("data/popular_rank.csv", index_col="MovieID")

    # Load similarity matrix
    S = pd.read_csv("data/S.csv", index_col=0)

    return R, movies, popular_rank, S

R, movies, popular_rank, S = load_data()

# -----------------------------
# IBCF Function
# -----------------------------
def myIBCF(newuser, S, popular_rank):
    predictions = {}
    rated_mask = ~newuser.isna()
    rated_movies = newuser.index[rated_mask]

    for i in S.index:
        if i in rated_movies:
            continue
        neighbors = S.loc[i].dropna()
        if neighbors.empty:
            continue
        rated_neighbors = neighbors.index.intersection(rated_movies)
        if rated_neighbors.empty:
            continue

        sim_vals = neighbors.loc[rated_neighbors]
        user_rated_values = newuser.loc[rated_neighbors]
        denom = sim_vals.sum()

        if denom == 0:
            continue

        pred = (sim_vals * user_rated_values).sum() / denom
        predictions[i] = pred

    pred_series = pd.Series(predictions).dropna().sort_values(ascending=False)

    if len(pred_series) < 10:
        needed = 10 - len(pred_series)
        exclude = rated_movies.union(pred_series.index)
        fill_candidates = [m for m in popular_rank.index if m not in exclude]
        fill = fill_candidates[:needed]
        final_recs = list(pred_series.index) + list(fill)
    else:
        final_recs = pred_series.index[:10].tolist()

    final_recs = [movie for movie in final_recs if movie in S.index and movie in movies.index]
    return final_recs

# -----------------------------
# Streamlit App
# -----------------------------

st.title("Movie Recommender System (IBCF)")

st.markdown("""
Welcome to the Movie Recommender!
**Instructions:**
1. Rate some of the sample movies below.
2. Click 'Get Recommendations'.
3. We'll show you the top 10 movies we think you'll enjoy!
""")

st.markdown("""
<style>
.movie-card img {
    max-width: 100%;
    border-radius: 10px;
    height: 200px; /* Ensures consistent image height */
    object-fit: cover; /* Scales image to fill the frame */
}
.movie-title {
    text-align: center;
    font-size: 14px;
    font-weight: bold;
    margin-top: 5px;
    min-height: 50px; /* Ensures consistent height for titles */
}
.movie-genres {
    text-align: center;
    font-size: 12px;
    color: gray;
    margin-bottom: 10px;
}
.slider-container {
    text-align: center;
    margin-top: 10px;
}
.movie-card {
    padding: 20px; /* Adds spacing around the entire card */
}
</style>
""", unsafe_allow_html=True)


# Path to the MovieImages folder
IMAGE_FOLDER = "MovieImages"

# Select a subset of movies to display for rating
sample_movies = popular_rank.index[:50]  # Top 50 popular movies
sample_df = movies.loc[sample_movies, ["Title", "Genres"]]

st.write("Please rate the following movies (1-5 stars). If you haven't seen a movie, leave it unrated.")

user_ratings = {}
def get_movie_image(mid):
    image_path = os.path.join(IMAGE_FOLDER, f"{mid[1:]}.jpg")
    if os.path.exists(image_path):
        return image_path
    return None

# Number of columns per row
columns_per_row = 3  # Adjust this value for more/less movies per row

# Display movies in rows of `columns_per_row`
for i in range(0, len(sample_movies), columns_per_row):
    row_movies = sample_movies[i:i+columns_per_row]
    cols = st.columns(columns_per_row)  # Create columns for the row

    for col, mid in zip(cols, row_movies):
        with col:
            image_path = os.path.join(IMAGE_FOLDER, f"{mid[1:]}.jpg")
            title = sample_df.loc[mid, "Title"]
            genres = sample_df.loc[mid, "Genres"]

            # Display the image
            if os.path.exists(image_path):
                col.image(image_path, use_container_width=True)
            else:
                col.write("[No Image Available]")

            # Display title
            col.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)

            # Display genres
            col.markdown(f'<div class="movie-genres">{genres}</div>', unsafe_allow_html=True)

            # Display the slider
            rating = col.slider(f"Rate {mid}", 0, 5, 0, key=mid, label_visibility="collapsed")
            user_ratings[mid] = rating if rating > 0 else np.nan


if st.button("Get Recommendations"):
    # Create a full rating vector with length = R.shape[1]
    all_movies = R.columns
    newuser = pd.Series(index=all_movies, data=np.nan)
    for m in user_ratings:
        if m in newuser.index:
            newuser[m] = user_ratings[m]

    # Call myIBCF to get recommendations
    recommendations = myIBCF(newuser, S, popular_rank)

    st.write("**Your Top 10 Recommendations:**")
    recs_df = movies.loc[recommendations, ["Title", "Genres"]].copy()
    recs_df["Title"] = recs_df["Title"].fillna("Unknown Title")
    recs_df["Genres"] = recs_df["Genres"].fillna("Unknown Genres")

    rec_columns = st.columns(5)
    for i, (idx, row) in enumerate(recs_df.iterrows()):
        rec_col = rec_columns[i % 5]

        # Display recommendation image
        image_path = os.path.join(IMAGE_FOLDER, f"{idx[1:]}.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            rec_col.image(image, use_container_width=True)
        else:
            rec_col.write("[No Image Available]")

        # Display movie title and genres
        rec_col.markdown(f"**{row['Title']}**")
        rec_col.caption(row["Genres"])

st.markdown("""
*Note: This is a demo app. If no ratings are provided or fewer than 10 predictions are generated, we fill the list with the most popular movies you haven't rated.*
""")
