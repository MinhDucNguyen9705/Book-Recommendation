import numpy as np
from model import ALS
import pandas as pd
class MatrixFactorizationInference:
    def __init__(self, weight_path, interaction_file, book_file):
        self.data = self.load_data(interaction_file, book_file)
        self.books = self.load_books(book_file)
        self.preprocess_data()
        self.model = ALS(self.data, k=3, lamu=0.1, lamb=0.1)
        self.model.load_weights(weight_path)

    def load_data(self, interaction_file, book_file):
        import pandas as pd
        ratings = pd.read_csv(interaction_file)
        finalbooks = pd.read_csv(book_file)
        return ratings

    def load_books(self, book_file):
        import pandas as pd
        return pd.read_csv(book_file)

    def preprocess_data(self):
        import pandas as pd
        useronly = self.data.groupby('user_id', as_index=False).agg({'rating': pd.Series.count}).sort_values('rating', ascending=False).head(15000)
        finalratings = self.data[self.data.user_id.isin(useronly.user_id)]
        bookonly = finalratings.groupby('book_id', as_index=False).agg({'rating': pd.Series.count}).sort_values('rating', ascending=False).head(8000)
        finalratings = finalratings[finalratings.book_id.isin(bookonly.book_id)]
        finalbooks = self.books[self.books.book_id.isin(bookonly.book_id)].reset_index(drop=True)
        finalbooks["newbookid"] = finalbooks.index + 1
        finalratings = finalratings.merge(finalbooks[['book_id', 'newbookid']], how='left', on=['book_id'])
        finalratings['newuser_id'] = finalratings.groupby('user_id').grouper.group_info[0] + 1
        self.data = finalratings
    def __init__(self, prediction_path = '../weights/matrix_factorization.weights'):
        self.pred_df = pd.read_csv(prediction_path)
    def get_user_book_pair_rating(self, user_id, book_id):
            match = self.pred_df[
                (self.pred_df['user_id'] == user_id) & 
                (self.pred_df['book_id'] == book_id)
            ]
            if not match.empty:
                return match.iloc[0]['pred']
            else:
                return None

    def get_top_k_recommendations(self, newuser_id, k=10, filter_read=True):
        user_ratings = self.get_user_ratings(newuser_id)
        # Sắp xếp giảm dần theo pred
        sorted_books = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
        # Top k
        return sorted_books[:k]

    def get_user_ratings(self, newuser_id):
       
        df = self.pred_df[self.pred_df['newuser_id'] == newuser_id]
        # chuyển thành dict
        return dict(zip(df['newbookid'], df['pred']))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Matrix Factorization Inference")
    parser.add_argument('--user_id', type=int, help='User ID (newuser_id)', required=False)
    parser.add_argument('--book_id', type=int, help='Book ID (newbookid)', required=False)
    parser.add_argument('--top_k', type=int, default=10, help='Number of top recommendations')
    args = parser.parse_args()

    weight_path = '../weights/matrix_factorization.weights'
    interaction_file = '../data/interaction.csv'
    book_file = '../data/final_books.csv'

    mfi = MatrixFactorizationInference(weight_path, interaction_file, book_file)

    if args.user_id and args.book_id:
        rating = mfi.get_user_book_pair_rating(args.user_id, args.book_id)
        print(f"Predicted rating for user {args.user_id} and book {args.book_id}: {rating}")
    elif args.user_id:
        top_k = mfi.get_top_k_recommendations(args.user_id, k=args.top_k)
        print(f"Top {args.top_k} recommendations for user {args.user_id}: {top_k}")
    else:
        print("Please provide at least --user_id.")
