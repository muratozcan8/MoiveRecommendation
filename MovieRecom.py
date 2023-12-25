import random
import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request

app = Flask(__name__)

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")


def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


movies["clean_title"] = movies["title"].apply(clean_title)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])


def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        user_movie_input = request.form["user_movie_input"]
        user_input_cleaned = clean_title(user_movie_input)
        found_movies = movies[movies['clean_title'].str.lower() == user_input_cleaned.lower()]
        if found_movies.empty:
            results.append({"message": "Movie not found in the dataset."})
        else:
            search_results = search(user_movie_input)
            movie_id = search_results.iloc[0]["movieId"]
            similar_movies = find_similar_movies(movie_id)
            results.extend(similar_movies[["score", "title", "genres"]].to_dict(orient="records"))

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
