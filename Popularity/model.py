import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class HybridModel:
    def __init__(self, n_factors=20, max_iter=100,
                 w_cf=0.6, w_cbf=0.3, w_pop=0.1):
        self.n_factors = n_factors
        self.nmf_model = NMF(
            n_components=self.n_factors,
            init='random',
            random_state=42,
            max_iter=max_iter
        )
        self.tfidf = TfidfVectorizer(max_features=5000,
                                     stop_words='english')
        self.author_encoder = OneHotEncoder(handle_unknown='ignore')
        self.popularity_scaler = MinMaxScaler()
        self.book_id_to_idx = {}
        self.W = None
        self.H = None
        self.popularity_dict = {}
        # weights for combining
        self.w_cf = w_cf
        self.w_cbf = w_cbf
        self.w_pop = w_pop

    def fit(self, books: pd.DataFrame, interactions: pd.DataFrame):
        # same as before
        self.user_book_matrix = (
            interactions.pivot_table(
                index='user_id', columns='book_id', values='rating'
            ).fillna(0)
        )
        W_arr = self.nmf_model.fit_transform(self.user_book_matrix)
        H_mat = self.nmf_model.components_
        self.W = np.asarray(W_arr)
        self.H = np.asarray(H_mat)

        books = books.copy()
        books['text'] = (books['genres'].fillna('') + ' ' +
                         books['description'].fillna('') + ' ' +
                         books['first_author'].fillna(''))
        tfidf_sparse = self.tfidf.fit_transform(books['text'])
        tfidf_mat = tfidf_sparse.toarray()
        author_ohe = self.author_encoder.fit_transform(
            books[['first_author']]
        ).toarray()
        self.tfidf_matrix = np.hstack([tfidf_mat, author_ohe])
        self.book_id_to_idx = {
            bid: idx for idx, bid in enumerate(books['book_id'])
        }
        pop = books[['book_id', 'ratings_count', 'average_rating']].copy()
        pop['pop_score'] = pop['ratings_count'] * pop['average_rating']
        pop['pop_norm'] = (self.popularity_scaler
                           .fit_transform(pop[['pop_score']])
                           .flatten())
        self.popularity_dict = dict(zip(pop['book_id'], pop['pop_norm']))
        self.books = books.set_index('book_id')

    def _build_prediction_matrix(self):
        # collaborative filtering matrix
        cf_pred = np.dot(self.W, self.H)
        cf_df = pd.DataFrame(
            cf_pred,
            index=self.user_book_matrix.index,
            columns=self.user_book_matrix.columns
        )
        # normalize CF per user
        cf_norm = cf_df.apply(
            lambda row: MinMaxScaler().fit_transform(row.values.reshape(-1,1)).flatten(),
            axis=1, result_type='broadcast'
        )
        cf_norm.index = cf_df.index; cf_norm.columns = cf_df.columns

        # content-based and popularity for all users
        cbf_list = []
        for uid in self.user_book_matrix.index:
            ratings = self.user_book_matrix.loc[uid]
            liked = ratings[ratings >= 4].index
            if len(liked)==0:
                sim = np.zeros(len(self.books))
            else:
                idxs = [self.book_id_to_idx[b] for b in liked
                        if b in self.book_id_to_idx]
                profile = self.tfidf_matrix[idxs].mean(axis=0)
                sim = cosine_similarity(
                    profile.reshape(1,-1), self.tfidf_matrix
                ).flatten()
            cbf_list.append(sim)
        cbf_mat = pd.DataFrame(
            cbf_list,
            index=self.user_book_matrix.index,
            columns=self.books.index
        )
        cbf_norm = cbf_mat.apply(
            lambda col: MinMaxScaler().fit_transform(col.values.reshape(-1,1)).flatten(),
            axis=0, result_type='broadcast'
        )
        cbf_norm.index = cbf_mat.index; cbf_norm.columns = cbf_mat.columns

        pop_vec = pd.Series(
            [self.popularity_dict.get(b,0) for b in self.books.index],
            index=self.books.index
        )

        # combine with weights
        final = (
            self.w_cf * cf_norm +
            self.w_cbf * cbf_norm +
            self.w_pop * pop_vec
        )
        self.pred_ratings_df = final

    def predict(self, X, params=None):
        if not hasattr(self, 'pred_ratings_df'):
            self._build_prediction_matrix()
        if isinstance(X, pd.DataFrame):
            user_ids = X['user_id'].tolist()
        elif isinstance(X,list):
            user_ids = X
        else:
            raise ValueError("Input must be DataFrame or list of user_id")
        results = []
        for uid in user_ids:
            recs = self.recommend(uid, top_n=5)
            results.append(recs.set_index('title')['score'].to_dict())
        return results

    # recommend() unchanged
    def recommend(self, user_id, top_n=10,
                  w_cf=None, w_cbf=None, w_pop=None):
        # use stored weights
        return self._recommend(user_id, top_n)
    def _recommend(self, user_id, top_n):
        if user_id not in self.pred_ratings_df.index:
            return pd.DataFrame()
        user_scores = self.pred_ratings_df.loc[user_id]
        seen = set(self.user_book_matrix.loc[user_id]
                   [self.user_book_matrix.loc[user_id]>0].index)
        candidates = user_scores[~user_scores.index.isin(seen)]
        top = candidates.sort_values(ascending=False).head(top_n)
        recs = self.books.loc[top.index,
            ['title','first_author','average_rating','ratings_count']
        ].copy()
        recs['score'] = top.values
        return recs.sort_values('score', ascending=False)


def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
