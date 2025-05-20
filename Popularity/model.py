import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import joblib

class CBFPopModel:
    def __init__(self, max_features=5000, w_cbf=0.3, w_pop=0.1, w_mf=0.6, 
                 n_factors=50, regularization=0.01, normalize_ratings=True):
        self.w_cbf = w_cbf
        self.w_pop = w_pop
        self.w_mf = w_mf
        self.n_factors = n_factors
        self.regularization = regularization
        self.normalize_ratings = normalize_ratings

        self.tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.pop_scaler = MinMaxScaler()
        self.rating_scaler = MinMaxScaler()

        self.books = None
        self.tfidf_matrix = None
        self.pop_dict = {}
        self.interactions = None
        self.pred_df = None
        
        # User and item biases
        self.user_biases = {}
        self.item_biases = {}
        self.global_mean = 0
        
        # Matrix factorization components
        self.user_factors = None
        self.item_factors = None
        self.user_id_map = {}
        self.book_id_map = {}
        self.reverse_user_map = {}
        self.reverse_book_map = {}

    def fit(self, books: pd.DataFrame, interactions: pd.DataFrame):
        self.interactions = interactions.copy()
        
        # Calculate global mean rating
        self.global_mean = interactions['rating'].mean()
        
        # Calculate user and item biases
        if self.normalize_ratings:
            self._calculate_biases()
        
        # Content-based filtering component
        books = books.copy()
        # Combine genres, title, and description for better content representation
        books['text'] = (
            books['genres'].fillna('') + ' ' + 
            books['title'].fillna('') + ' ' + 
            books['description'].fillna('')
        )
        tf = self.tfidf.fit_transform(books['text'])
        self.tfidf_matrix = tf.toarray()
        self.books = books.set_index('book_id')

        # Popularity component with improved formula
        pop = books[['book_id', 'ratings_count', 'average_rating']].copy()
        # Bayesian average to handle books with few ratings
        C = pop['ratings_count'].mean()
        m = pop['average_rating'].mean()
        pop['pop_raw'] = (pop['ratings_count'] * pop['average_rating'] + C * m) / (pop['ratings_count'] + C)
        pop['pop_norm'] = self.pop_scaler.fit_transform(pop[['pop_raw']]).flatten()
        self.pop_dict = dict(zip(pop['book_id'], pop['pop_norm']))
        
        # Matrix factorization component
        self._fit_matrix_factorization()

    def _calculate_biases(self):
        """Calculate user and item biases for rating normalization"""
        # Group by user_id and calculate mean rating per user
        user_means = self.interactions.groupby('user_id')['rating'].mean()
        # User bias is the difference from global mean
        self.user_biases = {user: mean - self.global_mean 
                           for user, mean in user_means.items()}
        
        # Calculate item biases
        # First adjust ratings by user bias
        adjusted_ratings = []
        for _, row in self.interactions.iterrows():
            user_id = row['user_id']
            rating = row['rating']
            user_bias = self.user_biases.get(user_id, 0)
            adjusted_ratings.append(rating - self.global_mean - user_bias)
        
        # Create a temporary dataframe with adjusted ratings
        temp_df = self.interactions.copy()
        temp_df['adjusted_rating'] = adjusted_ratings
        
        # Group by book_id and calculate mean adjusted rating
        item_means = temp_df.groupby('book_id')['adjusted_rating'].mean()
        self.item_biases = dict(item_means)

    def _fit_matrix_factorization(self):
        """Fit matrix factorization component using SVD"""
        # Create user and book ID mappings
        unique_users = self.interactions['user_id'].unique()
        unique_books = self.interactions['book_id'].unique()
        
        self.user_id_map = {user_id: i for i, user_id in enumerate(unique_users)}
        self.book_id_map = {book_id: i for i, book_id in enumerate(unique_books)}
        self.reverse_user_map = {i: user_id for user_id, i in self.user_id_map.items()}
        self.reverse_book_map = {i: book_id for book_id, i in self.book_id_map.items()}
        
        # Create user-item matrix
        rows, cols, data = [], [], []
        for _, row in self.interactions.iterrows():
            user_idx = self.user_id_map.get(row['user_id'])
            book_idx = self.book_id_map.get(row['book_id'])
            
            if user_idx is not None and book_idx is not None:
                rows.append(user_idx)
                cols.append(book_idx)
                
                # Apply bias correction if enabled
                if self.normalize_ratings:
                    user_bias = self.user_biases.get(row['user_id'], 0)
                    item_bias = self.item_biases.get(row['book_id'], 0)
                    adjusted_rating = row['rating'] - self.global_mean - user_bias - item_bias
                    data.append(adjusted_rating)
                else:
                    data.append(row['rating'])
        
        # Create sparse matrix
        n_users = len(unique_users)
        n_books = len(unique_books)
        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_books))
        
        # Apply SVD
        svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = svd.fit_transform(user_item_matrix)
        self.item_factors = svd.components_.T

    def build_prediction_matrix(self):
        users = self.interactions['user_id'].unique()
        books = list(self.books.index)
        data = []

        for u in users:
            # Content-based filtering component
            user_int = self.interactions[self.interactions['user_id']==u]
            liked = user_int[user_int['rating']>=4]['book_id'].tolist()
            if liked:
                idxs = [books.index(b) for b in liked if b in books]
                if idxs:  # Check if idxs is not empty
                    profile = self.tfidf_matrix[idxs].mean(axis=0).reshape(1, -1)
                    cbf_score = cosine_similarity(profile, self.tfidf_matrix).flatten()
                else:
                    cbf_score = np.zeros(len(books))
            else:
                cbf_score = np.zeros(len(books))

            # Popularity component
            pop_score = np.array([self.pop_dict.get(b, 0) for b in books])
            
            # Matrix factorization component
            mf_score = np.zeros(len(books))
            if u in self.user_id_map:
                user_idx = self.user_id_map[u]
                user_vec = self.user_factors[user_idx].reshape(1, -1)
                
                for i, book_id in enumerate(books):
                    if book_id in self.book_id_map:
                        book_idx = self.book_id_map[book_id]
                        book_vec = self.item_factors[book_idx].reshape(-1, 1)
                        mf_score[i] = user_vec @ book_vec
                
                # Normalize MF scores to 0-1 range
                if mf_score.max() > mf_score.min():
                    mf_score = (mf_score - mf_score.min()) / (mf_score.max() - mf_score.min())
            
            # Combine scores with weights
            final = (self.w_cbf * cbf_score + 
                     self.w_pop * pop_score + 
                     self.w_mf * mf_score)
            
            # Apply bias correction for final prediction
            if self.normalize_ratings:
                user_bias = self.user_biases.get(u, 0)
                final_with_bias = []
                
                for i, b in enumerate(books):
                    item_bias = self.item_biases.get(b, 0)
                    # Scale final score to rating range and add biases
                    pred = final[i] * 4 + 1  # Scale 0-1 to 1-5
                    pred = self.global_mean + user_bias + item_bias + (pred - 3)  # Adjust with biases
                    pred = max(1, min(5, pred))  # Clip to rating range
                    final_with_bias.append(pred)
                
                data.append(final_with_bias)
            else:
                # Scale to rating range without bias correction
                final = final * 4 + 1  # Scale 0-1 to 1-5
                data.append(final)

        self.pred_df = pd.DataFrame(
            np.vstack(data),
            index=users,
            columns=books
        )

    def evaluate_rmse(self, test_interactions=None):
        """
        Calculate RMSE on provided test interactions or self.interactions if None
        Returns RMSE value and number of predictions made
        """
        if test_interactions is None:
            test_interactions = self.interactions
            
        true_vals = []
        pred_vals = []
        
        for _, row in test_interactions.iterrows():
            u, b, r = row['user_id'], row['book_id'], row['rating']
            if u in self.pred_df.index and b in self.pred_df.columns:
                true_vals.append(r)
                pred_vals.append(self.pred_df.at[u, b])
        
        # Handle case where no predictions can be made
        if len(true_vals) == 0:
            return float('inf'), 0
            
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        return rmse, len(true_vals)

    def recommend(self, user_id: int, top_n=5):
        if self.pred_df is None:
            raise ValueError("You must call build_prediction_matrix() before recommend().")

        if user_id not in self.pred_df.index:
            return pd.DataFrame(columns=['book_id', 'score'])

        user_scores = self.pred_df.loc[user_id]
        seen = self.interactions[self.interactions['user_id'] == user_id]['book_id'].tolist()
        candidates = user_scores.drop(labels=seen, errors='ignore')

        top = candidates.sort_values(ascending=False).head(top_n)
        return top.reset_index().rename(columns={'index': 'book_id', user_id: 'score'})

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict top 5 book recommendations for each user_id in X.
        X must be a DataFrame with a column 'user_id'.
        """
        if self.pred_df is None:
            raise ValueError("You must call build_prediction_matrix() before predict().")

        results = []
        for user_id in X['user_id']:
            recs = self.recommend(user_id, top_n=5)
            recs['user_id'] = user_id
            results.append(recs)

        return pd.concat(results, ignore_index=True)

def save_model(m, path):
    joblib.dump(m, path)

def load_model(path):
    return joblib.load(path)
