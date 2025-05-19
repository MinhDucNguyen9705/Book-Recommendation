import numpy as np
import pandas as pd
import numpy as np
from model import ALS
from utils import load_data, preprocess_data, split_data
import argparse

def dcg_k(r, k):
    r = np.asarray(r, dtype=np.float64)[:k]
    return np.sum(2**r / np.log2(np.arange(2, r.size + 2)))

def ndcg_k(r, k):
    dcg_max = dcg_k(sorted(r, reverse=True), k)
    return dcg_k(r, k) / dcg_max if dcg_max else 0.

def mean_ndcg(rs):
    return np.mean([ndcg_k(r, len(r)) for r in rs])

def rmse(y, h):
    return np.sqrt(np.mean((y - h) ** 2))

class MatrixFactorizationTester:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data

    def evaluate(self,U,B,test_data):
        
        nR = np.zeros(len(test_data))
        for i in range(len(test_data)):
            nR[i] = B[:, test_data.newbookid.iloc[i] - 1] @ U[test_data.newuser_id.iloc[i] - 1, :]
    
        rmse_score = rmse(nR, self.test_data.rating)
        user_ids = self.test_data['newuser_id'].unique()
        mflist = [self.test_data[self.test_data.newuser_id == uid]['rating'].tolist() for uid in user_ids]
        ndcg_score = mean_ndcg(mflist)
        return rmse_score, ndcg_score

if __name__ == "__main__":
    ratings, books = load_data('../data/interaction.csv', '../data/final_books.csv')
    finalratings, finalbooks = preprocess_data(ratings, books)
    train, test = split_data(finalratings)
    test_data = test
    parser = argparse.ArgumentParser()
    parser.add_argument('--lamu', type=float, default=0.1, help='Regularization for users')
    parser.add_argument('--lamb', type=float, default=0.1, help='Regularization for books')
    parser.add_argument('--k', type=int, default=10, help='Number of latent factors')
    args = parser.parse_args()
    
    model = ALS(test, k=args.k, lamu=args.lamu, lamb=args.lamb)
    w = np.load("../weights/MF.weights.h5.npz")
    U = w['U']
    B = w['B']
    R = U.dot(B)
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
# train and evaluate model here
    tester = MatrixFactorizationTester(model, test_data)
    rmse, ndcg = tester.evaluate(U,B,test_data)
    print(f"RMSE: {rmse}, NDCG: {ndcg}")
