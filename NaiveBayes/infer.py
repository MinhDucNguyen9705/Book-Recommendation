import pickle
import pandas as pd
from preprocessing_data import preprocess_text, vectorize_text

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
    with open('/Users/khangdoan/Documents/Git/Book-Recommendation/NaiveBayes/his.pkl', 'rb') as f:
        his = pickle.load(f)
    with open('/Users/khangdoan/Documents/Git/Book-Recommendation/NaiveBayes/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    path = '/Users/khangdoan/Library/CloudStorage/OneDrive-HanoiUniversityofScienceandTechnology/[HUST] General Subjects/2024.2/IT3190E - Machine Learning/mini-prj/'
    books = pd.read_csv(path + 'final_books.csv')
    ratings = pd.read_csv(path + 'interaction.csv')
    # predited = predict_books(14, books, ratings)
    # print(books[books['book_id'] == 14854])
    book_id = 14854
    user_id = 14
    # print(predict_single_book(user_id, book_id, books, ratings, his, vocab))
    print(predict_books(user_id, books, ratings, his, vocab))