from content_based import recommend

movie_name = input("Enter Movie Name: ")

results = recommend(movie_name)

print("\nRecommended Movies:\n")

for movie in results:
    print(movie)