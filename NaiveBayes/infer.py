import pickle
import pandas as pd
from preprocessing_data import preprocess_text, vectorize_text
from joblib import load

def predict_books(user_id, books, ratings, his, vocab):
    rate = {}
    for book_id in books['book_id'].unique():
        if ratings[(ratings['user_id']== user_id) & (ratings['book_id']== book_id)].shape[0] > 0:
            rate[book_id] = ratings[ratings['book_id'] == book_id][ratings['user_id'] == user_id].rating.values[0]
        else:
            processed = preprocess_text(books[books['book_id'] == book_id]['description'].values[0])
            vec = vectorize_text(processed, vocab)
            rate[book_id] = his[user_id].predict([vec])[0]
    return rate

def predict_single_book(user_id, book_id, books, ratings, his, vocab):
    if ratings[(ratings['user_id']== user_id) & (ratings['book_id']== book_id)].shape[0] > 0:
        return ratings[ratings['book_id'] == book_id][ratings['user_id'] == user_id].rating.values[0]
    else:
        processed = preprocess_text(books[books['book_id'] == book_id]['description'].values[0])
        vec = vectorize_text(processed, vocab)
        return his[user_id].predict([vec])[0]

if __name__ == "__main__":
    his1 = load("/Users/khangdoan/Documents/Git/Book-Recommendation/NaiveBayes/his1.joblib")
    his2 = load("/Users/khangdoan/Documents/Git/Book-Recommendation/NaiveBayes/his2.joblib")
    his = {**his1, **his2}
    vocab = load("/Users/khangdoan/Documents/Git/Book-Recommendation/NaiveBayes/vocab.joblib")

    path = '/Users/khangdoan/Documents/Git/Book-Recommendation/Data/'
    books = pd.read_csv(path + 'final_books.csv')
    ratings = pd.read_csv(path + 'interaction.csv')
    # predited = predict_books(14, books, ratings)
    # print(books[books['book_id'] == 14854])
    book_id = 14854
    user_id = 14
    # print(predict_single_book(user_id, book_id, books, ratings, his, vocab))
    print(predict_books(user_id, books, ratings, his, vocab))