
from utils import get_words, rmse, dcg_k, ndcg_k, mean_ndcg
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
class TfIdf:
    def __init__(self, finalbooks, finalratings, train=True):
        self.finalbooks = finalbooks
        self.finalratings = finalratings
        if train:
            self.SimC, self.book_idx = self.create_tf_matrix()
    def create_tf_matrix(self):
        word_counts = Counter()

        for txt in self.finalbooks['text']:
            for w in set(get_words(txt)):
                word_counts[w] += 1

        filtered = [
            w for w,c in word_counts.items()
            if c >= 10
            and w not in stopwords.words('english')
            and len(w) > 1
        ]

        vocab = { w: i for i, w in enumerate(filtered) }

        A = np.zeros((len(self.finalbooks), len(vocab)), dtype=np.float32)
        for i, txt in enumerate(self.finalbooks['text']):
            for w in get_words(txt):
                if w in vocab:
                    A[i, vocab[w]] += 1
        
        TF  = A / (A.sum(axis=1,keepdims=True)+1e-9)

        DF  = np.count_nonzero(A>0, axis=0)

        IDF = np.log(len(self.finalbooks) / (DF+1e-9))
        TFIDF = TF * IDF

        TFIDF = TFIDF / (np.linalg.norm(TFIDF, axis=1, keepdims=True) + 1e-9)

        SimC = TFIDF @ TFIDF.T
        np.fill_diagonal(SimC, 1.0)

        book_idx = {bid:i for i,bid in enumerate(self.finalbooks['book_id'])}

        return SimC, book_idx
        
    def predict_all_books_single_user(self, user_id, SimC, book_idx):
        user_data = self.finalratings[self.finalratings['newuser_id'] == user_id]
        user_data = user_data[user_data['newbookid'].isin(book_idx)]  # chỉ lấy sách hợp lệ

        if user_data.empty:
            raise ValueError("User has no valid ratings for prediction.")

        bidx = [book_idx[b] for b in user_data['newbookid']] #lấy book
        true_ratings = user_data['rating'].values

        # Tính độ tương đồng giữa tất cả sách và các sách mà người dùng đã đánh giá
        sim_mat = SimC[:, bidx]  # shape: (n_books, len(bidx))

        # Tính điểm dự đoán cho tất cả sách
        preds = (sim_mat * true_ratings[np.newaxis, :]).sum(axis=1) / (sim_mat.sum(axis=1) + 1e-9)

        # Trả về DataFrame với tất cả sách và dự đoán tương ứng
        return pd.DataFrame({
            'newuser_id': user_id,
            'newbookid': self.finalbooks['newbookid'],
            'pred': preds
        })
    
    def predict_single_book_single_user(self, user_id, book_id, SimC, book_idx):
        res = self.predict_all_books_single_user(user_id, SimC, book_idx)
        return res[res['newbookid'] == book_id]
    
    def predict_for_eval(self, test_df):
        all_preds = []
        for uid, grp in test_df.groupby('newuser_id'):
            grp_sorted = grp.sort_values('newbookid')
            filtered = grp_sorted[grp_sorted['newbookid'].isin(self.book_idx)]
            bidx = [self.book_idx[b] for b in filtered['newbookid']]
            if not bidx:  # skip if empty
                continue
            true_r = filtered['rating'].values
            sim_mat = self.SimC[:, bidx]  # shape: (n_books, len(bidx))
            preds   = (sim_mat * true_r[np.newaxis, :]).sum(axis=1) / (sim_mat.sum(axis=1) + 1e-9)
            for bi, pred in enumerate(preds):
                all_preds.append((uid, self.finalbooks.at[bi, 'newbookid'], pred))
        return pd.DataFrame(all_preds, columns=['newuser_id','newbookid','pred'])
        
    
    def eval_metrics(self, test_df):
        pred_test = self.predict_for_eval(test_df)
        merged_te = test_df.merge(pred_test , on=['newuser_id','newbookid'])

        print("Test  RMSE:", np.round(rmse(merged_te['rating'], merged_te['pred']),3))

        ndcg_te = [ ndcg_k(grp.sort_values('pred')['rating'].tolist(),
                        len(grp))
                    for _,grp in merged_te.groupby('newuser_id') ]

        print("Test  mean NDCG:", np.round(np.mean(ndcg_te),3))