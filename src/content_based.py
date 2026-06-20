from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_data

movies = preprocess_data()

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["content"])

similarity_matrix = cosine_similarity(tfidf_matrix)
def recommend(movie_title, top_n=10):

    matches = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            na=False
        )
    ]

    if matches.empty:
        return ["Movie not found"]

    movie_index = matches.index[0]

    scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for idx, score in scores[1:top_n+1]:
        recommendations.append(
            movies.iloc[idx]["title"]
        )

    return recommendations