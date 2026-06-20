import pandas as pd

def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    tags = pd.read_csv("data/tags.csv")
    links = pd.read_csv("data/links.csv")

    return movies, ratings, tags, links