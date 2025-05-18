import pandas as pd
import numpy as np

def recommend_single_book(user_id, book_id, books, ratings):
    if ratings[(ratings['user_id']== user_id) & (ratings['book_id']== book_id)].shape[0] > 0:
        return ratings[ratings['book_id'] == book_id][ratings['user_id'] == user_id].rating.values[0]
    else:
        return books[books['book_id'] == book_id]['average_rating'].values[0]

def recommend_books(user_id, books, ratings):
    rate = {}
    for book_id in books['book_id'].unique():
        if ratings[(ratings['user_id']== user_id) & (ratings['book_id']== book_id)].shape[0] > 0:
            rate[book_id] = ratings[ratings['book_id'] == book_id][ratings['user_id'] == user_id].rating.values[0]
        else:
            rate[book_id] = books[books['book_id'] == book_id]['average_rating'].values[0]
    return rate

if __name__ == "__main__":
    user_id = 2142
    path = '/Users/khangdoan/Documents/Git/Book-Recommendation/Data/'
    books = pd.read_csv(path + 'final_books.csv')
    ratings = pd.read_csv(path + 'interaction.csv')
    print(recommend_books(user_id, books, ratings))