import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(interaction_file, book_file):
    ratings = pd.read_csv(interaction_file)
    finalbooks = pd.read_csv(book_file)
    return ratings, finalbooks

def preprocess_data(ratings, books):
    useronly = ratings.groupby('user_id', as_index=False).agg({'rating': pd.Series.count}).sort_values('rating', ascending=False).head(15000)
    finalratings = ratings[ratings.user_id.isin(useronly.user_id)]
    bookonly = finalratings.groupby('book_id', as_index=False).agg({'rating': pd.Series.count}).sort_values('rating', ascending=False).head(8000)
    finalratings = finalratings[finalratings.book_id.isin(bookonly.book_id)]
    finalbooks = books[books.book_id.isin(bookonly.book_id)].reset_index(drop=True)
    finalbooks["newbookid"] = finalbooks.index + 1
    finalratings = finalratings.merge(finalbooks[['book_id', 'newbookid']], how='left', on=['book_id'])
    finalratings['newuser_id'] = finalratings.groupby('user_id').grouper.group_info[0] + 1
    return finalratings, finalbooks

def split_data(data, test_size=0.2, random_state=42):
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    train_user_counts = train['newuser_id'].value_counts()
    valid_train_users = train_user_counts[train_user_counts >= 2].index
    train = train[train['newuser_id'].isin(valid_train_users)]
    return train, test
def rmse(y, h):
    return np.sqrt(np.mean((y - h) ** 2))

if __name__ == "__main__":
    ratings, books = load_data('../data/interaction.csv', '../data/final_books.csv')
    processed_ratings, processed_books = preprocess_data(ratings, books)
    print(processed_ratings.head())