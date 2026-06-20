import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load data
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

ratings_small = ratings[["userId", "movieId", "rating"]]

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings_small,
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

model = SVD()

model.fit(trainset)

print("Model trained!")

# Example user
user_id = 1

# Movies user has not rated
rated_movies = ratings[
    ratings["userId"] == user_id
]["movieId"].tolist()

unrated_movies = movies[
    ~movies["movieId"].isin(rated_movies)
]

predictions = []

for movie_id in unrated_movies["movieId"]:

    pred = model.predict(
        user_id,
        movie_id
    )

    predictions.append(
        (movie_id, pred.est)
    )

# Top 10 recommendations
predictions.sort(
    key=lambda x: x[1],
    reverse=True
)

top_movies = predictions[:10]

print("\nTop Recommendations:\n")

for movie_id, score in top_movies:

    title = movies[
        movies["movieId"] == movie_id
    ]["title"].values[0]

    print(
        f"{title}  -> Predicted Rating: {score:.2f}"
    )