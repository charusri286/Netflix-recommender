import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD DATA
# =========================

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")

# =========================
# RATINGS
# =========================

avg_ratings = ratings.groupby("movieId")["rating"].mean()

movies = movies.merge(
    avg_ratings,
    on="movieId",
    how="left"
)

movies.rename(
    columns={"rating": "avg_rating"},
    inplace=True
)

movies["avg_rating"] = movies["avg_rating"].fillna(0)

# =========================
# TAGS
# =========================

movie_tags = tags.groupby("movieId")["tag"].apply(
    lambda x: " ".join(x.astype(str))
).reset_index()

movies = movies.merge(
    movie_tags,
    on="movieId",
    how="left"
)

movies["tag"] = movies["tag"].fillna("")

# =========================
# FEATURES
# =========================

movies["features"] = (
    movies["genres"]
    .fillna("")
    .str.replace("|", " ", regex=False)
    + " "
    + movies["tag"]
)

# =========================
# VECTORIZATION
# =========================

cv = TfidfVectorizer(stop_words="english")

feature_matrix = cv.fit_transform(
    movies["features"]
)

similarity = cosine_similarity(
    feature_matrix
)

# =========================
# RECOMMENDER
# =========================

def recommend_movies(movie_name):

    movie_name = movie_name.lower().strip()

    # --------------------------------
    # Franchise / title matches
    # --------------------------------

    title_matches = movies[
        movies["title"].str.contains(
            movie_name,
            case=False,
            na=False
        )
    ]

    if len(title_matches) > 1:

        result = []

        for _, row in title_matches.head(5).iterrows():

            result.append({
                "title": row["title"],
                "rating": round(row["avg_rating"], 1)
            })

        return result

    movie_indices = title_matches.index

    # --------------------------------
    # Movie not found
    # --------------------------------

    if len(movie_indices) == 0:

        suggestions = difflib.get_close_matches(
            movie_name,
            movies["title"].tolist(),
            n=5,
            cutoff=0.4
        )

        if suggestions:

            return [
                {
                    "title": s,
                    "rating": ""
                }
                for s in suggestions
            ]

        return [
            {
                "title": "Movie not found",
                "rating": ""
            }
        ]

    # --------------------------------
    # Similarity recommendations
    # --------------------------------

    idx = movie_indices[0]

    recommendations = []

    for i, sim_score in enumerate(similarity[idx]):

        if i == idx:
            continue

        rating_score = (
            movies.iloc[i]["avg_rating"] / 5.0
        )

        final_score = (
            0.7 * sim_score
            + 0.3 * rating_score
        )

        recommendations.append(
            (
                final_score,
                movies.iloc[i]["title"]
            )
        )

    recommendations.sort(
        key=lambda x: x[0],
        reverse=True
    )

    top_movies = []

    for _, movie_title in recommendations[:5]:

        rating = movies[
            movies["title"] == movie_title
        ]["avg_rating"].values[0]

        top_movies.append({
            "title": movie_title,
            "rating": round(rating, 1)
        })

    return top_movies