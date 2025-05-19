import pandas as pd
import numpy as np
from model import ALS
from utils import load_data, preprocess_data, split_data
import argparse

def train_model(data, book, k=3, lamu=0.1, lamb=0.1):
    ratings, books = data, book
    finalratings, finalbooks = preprocess_data(ratings, books)

    train, test = split_data(finalratings)
    # print(train.head())
    model = ALS(data=train, k=args.k, lamu=args.lamu, lamb=args.lamb)
    w = model.fit()
    np.savez("../weights/MF.weights.h5", U=w['U'], B=w['B'])

    R = w['U'].dot(w['B'])
    n_users, n_books = R.shape  # số user và book thật

    rflat = np.matrix.flatten(R)
    testy = np.repeat(np.arange(1, n_users + 1), n_books)
    testy = np.sort(testy)
    booky = np.tile(np.arange(1, n_books + 1), n_users)
    predictions = pd.DataFrame(np.column_stack((testy, booky, rflat)), 
                         columns=('newuser_id', 'newbookid', 'pred'))

    # Tạo ánh xạ newuser_id -> user_id
    user_mapping = finalratings[['newuser_id', 'user_id']].drop_duplicates().set_index('newuser_id')

    # Tạo ánh xạ newbookid -> book_id
    book_mapping = finalbooks[['newbookid', 'book_id']].set_index('newbookid')
    predictions = predictions.merge(user_mapping, how='left', left_on='newuser_id', right_index=True)
    predictions = predictions.merge(book_mapping, how='left', left_on='newbookid', right_index=True)

# Sắp xếp lại cột cho rõ ràng
    predictions = predictions[['user_id', 'book_id', 'newuser_id', 'newbookid', 'pred']]
    return predictions, test

if __name__ == "__main__":
    import pandas as pd
    # --- Thêm argparse để nhận từ dòng lệnh ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--lamu', type=float, default=0.1, help='Regularization for users')
    parser.add_argument('--lamb', type=float, default=0.1, help='Regularization for books')
    parser.add_argument('--k', type=int, default=10, help='Number of latent factors')
    args = parser.parse_args()

    ratings, books = load_data('../data/interaction.csv', '../data/final_books.csv')
    predictions = train_model(ratings, books)
    # print(predictions)
#     import pandas as pd

# # Tạo DataFrame từ tuple
#     columns = ['user_id', 'book_id', 'newuser_id', 'newbookid', 'pred']

#     df = pd.DataFrame(predictions, columns=columns)

# # Ghi ra CSV
#     df.to_csv("../weights/matrix_factorization.weights", index=False)
