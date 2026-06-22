from flask import Flask, render_template, request
from recommender import recommend_movies

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    searched_movie = None
    if request.method == "POST":

        searched_movie = request.form["movie"]

        recommendations = recommend_movies(
            searched_movie
        )

    return render_template(
        "index.html",
        recommendations=recommendations,
        searched_movie=searched_movie
    )

if __name__ == "__main__":
    app.run(debug=True)