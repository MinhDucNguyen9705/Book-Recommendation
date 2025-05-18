from preprocessing_data import preprocess
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from metrics import *
import os
from joblib import dump, load

def train_model(file_path):
    data, vocab = preprocess(file_path)
    
    train_df, test_df = train_test_split(data, stratify=data['user_id'], test_size=0.2, random_state=42)
    # train_df, val_df = train_test_split(train_df,
    #                            stratify=train_df['user_id'], 
    #                            test_size=0.1875,
    #                            random_state=42)
    
    history = {}
    track = 0
    for user_id in data['user_id'].unique():

        train = train_df[train_df['user_id'] == user_id]
        # val = val_df
        test = test_df[test_df['user_id'] == user_id]

        X_train = np.array(train['word_vector'].tolist())
        y_train = np.array(train['rating'].tolist())

        # X_test = np.array(test['word_vector'].tolist())
        # y_test = np.array(test['rating'].tolist())

        # X_val = np.array(val['word_vector'].tolist())
        # y_val = np.array(val['rating'].tolist())

        model = MultinomialNB(alpha=1)
        model.fit(X_train, y_train)
        # predictions = model.predict(X_test)

        history[user_id] = model
        # print(f"User {user_id} - RMSE: {rmse(predictions, y_test)}")
        track += 1
        if track % 1000 == 0:
            print(f"Trained {track} users")
    return history, vocab

if __name__ == "__main__":
    file_path = '/Users/khangdoan/Documents/Git/Book-Recommendation/Data/'
    history, vocab= train_model(file_path)

    his_path  = '/Users/khangdoan/Documents/Git/Book-Recommendation/NaiveBayes/his.joblib'
    vocab_path = '/Users/khangdoan/Documents/Git/Book-Recommendation/NaiveBayes/vocab.joblib'

    dump(history, his_path, compress=('lz4', 3))
    dump(vocab, vocab_path, compress=('lz4', 3))

    # with open(his_path, 'wb') as f:
    #     pickle.dump(history, f)
    # with open(vocab_path, 'wb') as f:
    #     pickle.dump(vocab, f)
    print("Models saved successfully.")