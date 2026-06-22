import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

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

print(
    movies[
        ["title", "avg_rating"]
    ].head(10)
)