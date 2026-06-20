from data_loader import load_data

def preprocess_data():
    movies, ratings, tags, links = load_data()

    tags_grouped = tags.groupby("movieId")["tag"].apply(
        lambda x: " ".join(x.astype(str))
    ).reset_index()

    movies = movies.merge(tags_grouped, on="movieId", how="left")

    movies["tag"] = movies["tag"].fillna("")

    movies["content"] = (
        movies["genres"].str.replace("|", " ", regex=False)
        + " "
        + movies["tag"]
    )

    return movies