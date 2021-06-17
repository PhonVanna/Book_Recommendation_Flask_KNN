import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, redirect, url_for, render_template


app = Flask(__name__)

book_file = 'static/data/BX-Books.csv'
user_file = 'static/data/BX-Users.csv'
rating_file = 'static/data/BX-Book-Ratings.csv'


def get_data(book_file, user_file, rating_file, popularity_threshold):
    popularity_threshold = popularity_threshold
    books = pd.read_csv(book_file, sep=';', error_bad_lines=False, encoding="latin-1")
    users = pd.read_csv(user_file, sep=';', error_bad_lines=False, encoding="latin-1")
    ratings = pd.read_csv(rating_file, sep=';', error_bad_lines=False, encoding="latin-1")
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    users.columns = ['userID', 'Location', 'Age']
    ratings.columns = ['userID', 'ISBN', 'bookRating']
    combine_book_rating = pd.merge(ratings, books, on='ISBN')
    columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

    book_ratingCount = (combine_book_rating.
        groupby(by = ['bookTitle'])['bookRating'].
        count().
        reset_index().
        rename(columns = {'bookRating': 'totalRatingCount'})
        [['bookTitle', 'totalRatingCount']]
    )

    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')
    us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
    us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
    us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
    return us_canada_user_rating_pivot


def get_matrix(us_canada_user_rating_pivot):
    us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)
    return us_canada_user_rating_matrix

us_canada_user_rating_pivot = get_data(book_file, user_file, rating_file, popularity_threshold=50)
us_canada_user_rating_matrix = get_matrix(us_canada_user_rating_pivot)


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(us_canada_user_rating_matrix)
    titles = []
    titles = [title for title in us_canada_user_rating_pivot.index]
    counts = [count for count in range(0, len(us_canada_user_rating_pivot))]


    count = -1
    rec_result = []
    if request.method == 'POST':
        query_index = int(request.form.get("content_id"))
        distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 10)
        us_canada_user_rating_pivot.iloc[query_index,:].values.reshape(1,-1)
        search_term = us_canada_user_rating_pivot.index[query_index]
        rec_result = [i for i in range(0, len(distances.flatten()))]
        rec_result = us_canada_user_rating_pivot.index[indices.flatten()[rec_result]]
        return render_template('home.html', titles=titles, counts=counts, count=count, search_term=search_term, rec_result=rec_result)

    return render_template('home.html', titles=titles, counts=counts, count=count, search_term='', rec_result=[])
    


if __name__ == '__main__':
    app.run(debug=True)