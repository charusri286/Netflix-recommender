import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("MOVIES")
print(movies.head())

print("\nRATINGS")
print(ratings.head())