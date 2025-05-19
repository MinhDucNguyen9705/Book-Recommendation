import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class HybridModel:
    def __init__(self, n_factors=20, max_iter=100):
        self.n_factors = n_factors
        self.nmf_model = NMF(n_components=self.n_factors, init='random', random_state=42, max_iter=max_iter)
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.popularity_scaler = MinMaxScaler()
        self.book_id_to_idx = {}
        self.W = None
        self.H = None
        self.popularity_dict = {}

    def fit(self, books: pd.DataFrame, interactions: pd.DataFrame):
        self.user_book_matrix = interactions.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
        self.W = self.nmf_model.fit_transform(self.user_book_matrix)
        self.H = self.nmf_model.components_

        books['text'] = books['genres'].fillna('') + ' ' + books['description'].fillna('')
        self.tfidf_matrix = self.tfidf.fit_transform(books['text'])
        self.book_id_to_idx = {bid: idx for idx, bid in enumerate(books['book_id'])}

        popularity = books[['ratings_count', 'average_rating']].copy()
        popularity['pop_score'] = popularity['ratings_count'] * popularity['average_rating']
        popularity['pop_score_norm'] = self.popularity_scaler.fit_transform(popularity[['pop_score']])
        self.popularity_dict = dict(zip(books['book_id'], popularity['pop_score_norm']))

        self.books = books.set_index('book_id')

    def predict(self):
        pred_ratings = np.dot(self.W, self.H)
        self.pred_ratings_df = pd.DataFrame(pred_ratings, index=self.user_book_matrix.index, columns=self.user_book_matrix.columns)

    def recommend(self, user_id, top_n=10, w_cf=0.6, w_cbf=0.3, w_pop=0.1):
        if user_id not in self.user_book_matrix.index:
            print(f"User {user_id} not found")
            return pd.DataFrame()

        cf_scores = self.pred_ratings_df.loc[user_id]
        user_ratings = self.user_book_matrix.loc[user_id]
        liked_books = user_ratings[user_ratings >= 4].index.intersection(self.books.index)

        if len(liked_books) == 0:
            cbf_scores = pd.Series(0, index=self.books.index)
        else:
            liked_indices = [self.book_id_to_idx[b] for b in liked_books if b in self.book_id_to_idx]
            user_profile = self.tfidf_matrix[liked_indices].mean(axis=0)
            cosine_sim = cosine_similarity(user_profile, self.tfidf_matrix).flatten()
            cbf_scores = pd.Series(cosine_sim, index=self.books.index)

        scaler_ = MinMaxScaler()
        cf_norm = pd.Series(scaler_.fit_transform(cf_scores.values.reshape(-1, 1)).flatten(), index=cf_scores.index)
        cbf_norm = pd.Series(scaler_.fit_transform(cbf_scores.values.reshape(-1, 1)).flatten(), index=cbf_scores.index)
        pop_scores = pd.Series([self.popularity_dict.get(b, 0) for b in self.books.index], index=self.books.index)

        final_scores = w_cf * cf_norm + w_cbf * cbf_norm + w_pop * pop_scores
        rated_books = set(user_ratings[user_ratings > 0].index)
        candidates = final_scores[~final_scores.index.isin(rated_books)]

        top_books = candidates.sort_values(ascending=False).head(top_n)
        recs = self.books.loc[top_books.index, ['title', 'first_author', 'average_rating', 'ratings_count']].copy()
        recs['score'] = top_books.values
        return recs.sort_values('score', ascending=False)

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
