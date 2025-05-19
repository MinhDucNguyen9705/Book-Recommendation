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
        self.nmf_model = NMF(
            n_components=self.n_factors,
            init='random',
            random_state=42,
            max_iter=max_iter
        )
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.popularity_scaler = MinMaxScaler()
        self.book_id_to_idx = {}
        self.W = None
        self.H = None
        self.popularity_dict = {}

    def fit(self, books: pd.DataFrame, interactions: pd.DataFrame):
        self.user_book_matrix = (
            interactions
            .pivot_table(
                index='user_id',
                columns='book_id',
                values='rating'
            )
            .fillna(0)
        )
        # Factorize using NMF
        W_arr = self.nmf_model.fit_transform(self.user_book_matrix)
        H_mat = self.nmf_model.components_
        # Ensure numpy arrays (not np.matrix)
        self.W = np.asarray(W_arr)
        self.H = np.asarray(H_mat)

        # Content-based: TF-IDF on genres + description
        books['text'] = (
            books['genres'].fillna('') + ' ' + books['description'].fillna('')
        )
        tfidf_sparse = self.tfidf.fit_transform(books['text'])
        self.tfidf_matrix = np.asarray(tfidf_sparse.toarray())
        self.book_id_to_idx = {bid: idx for idx, bid in enumerate(books['book_id'])}

        # Popularity score normalization
        popularity = books[['book_id', 'ratings_count', 'average_rating']].copy()
        popularity['pop_score'] = (
            popularity['ratings_count'] * popularity['average_rating']
        )
        popularity['pop_score_norm'] = (
            self.popularity_scaler
            .fit_transform(popularity[['pop_score']])
            .flatten()
        )
        self.popularity_dict = dict(
            zip(popularity['book_id'], popularity['pop_score_norm'])
        )

        self.books = books.set_index('book_id')

    def _build_prediction_matrix(self):
        pred = np.dot(self.W, self.H)
        self.pred_ratings_df = pd.DataFrame(
            pred,
            index=self.user_book_matrix.index,
            columns=self.user_book_matrix.columns
        )

    def predict(self, X, params=None):
        """
        MLflow scoring: predict(data, params=params)
        """
        if not hasattr(self, 'pred_ratings_df'):
            self._build_prediction_matrix()

        if isinstance(X, pd.DataFrame):
            user_ids = X['user_id'].tolist()
        elif isinstance(X, list):
            user_ids = X
        else:
            raise ValueError("Unsupported input format: must be DataFrame or list of user_id")

        results = []
        for uid in user_ids:
            recs = self.recommend(uid, top_n=5)
            results.append(recs.set_index('title')['score'].to_dict())
        return results

    def recommend(self, user_id, top_n=10, w_cf=0.6, w_cbf=0.3, w_pop=0.1):
        if user_id not in self.user_book_matrix.index:
            return pd.DataFrame(columns=['title','first_author','average_rating','ratings_count','score'])

        cf_scores = self.pred_ratings_df.loc[user_id]

        user_ratings = self.user_book_matrix.loc[user_id]
        liked = user_ratings[user_ratings >= 4].index
        if len(liked) == 0:
            cbf_scores = pd.Series(0, index=self.books.index)
        else:
            idxs = [self.book_id_to_idx[b] for b in liked if b in self.book_id_to_idx]
            user_profile = np.asarray(self.tfidf_matrix[idxs]).mean(axis=0)
            cosine_sim = cosine_similarity(
                user_profile.reshape(1, -1),
                self.tfidf_matrix
            ).flatten()
            cbf_scores = pd.Series(cosine_sim, index=self.books.index)

        # Normalize scores
        scaler = MinMaxScaler()
        cf_norm = pd.Series(
            scaler.fit_transform(cf_scores.values.reshape(-1,1)).flatten(),
            index=cf_scores.index
        )
        cbf_norm = pd.Series(
            scaler.fit_transform(cbf_scores.values.reshape(-1,1)).flatten(),
            index=cbf_scores.index
        )
        pop_scores = pd.Series(
            [self.popularity_dict.get(b,0) for b in self.books.index],
            index=self.books.index
        )

        # Weighted sum
        final = w_cf*cf_norm + w_cbf*cbf_norm + w_pop*pop_scores
        # Exclude already rated
        seen = set(user_ratings[user_ratings>0].index)
        candidates = final[~final.index.isin(seen)]

        # Top-n
        top = candidates.sort_values(ascending=False).head(top_n)
        recs = self.books.loc[top.index, [
            'title','first_author','average_rating','ratings_count'
        ]].copy()
        recs['score'] = top.values
        return recs.sort_values('score', ascending=False)


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
