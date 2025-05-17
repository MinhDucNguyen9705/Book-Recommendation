import pandas as pd
import numpy as np
import os 
import json
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(interaction_file, book_file):
    ratings = pd.read_csv(interaction_file, index_col=0)
    books = pd.read_csv(book_file, index_col=0)
    ratings = ratings.merge(books, on='book_id')
    return ratings, books

def preprocess_data(ratings, books):
    user_encoder = LabelEncoder()
    book_encoder = LabelEncoder()
    genre_encoder = OneHotEncoder(sparse_output=False)

    ratings['user'] = user_encoder.fit_transform(ratings['user_id'])
    ratings['book'] = book_encoder.fit_transform(ratings['book_id'])
    genre_encoded = genre_encoder.fit_transform(ratings[['most_tagged']])

    genre_columns = genre_encoder.get_feature_names_out(['most_tagged'])
    genre_df = pd.DataFrame(genre_encoded, columns=genre_columns, index=ratings.index)
    ratings = pd.concat([ratings.drop(columns=['most_tagged']), genre_df], axis=1)

    genre_encoded = genre_encoder.transform(books[['most_tagged']])
    genre_columns = genre_encoder.get_feature_names_out(['most_tagged'])
    genre_df = pd.DataFrame(genre_encoded, columns=genre_columns, index=books.index)
    books = pd.concat([books.drop(columns=['most_tagged']), genre_df], axis=1)

    return ratings, books, user_encoder, book_encoder, genre_encoder, genre_columns

def split_data(data, genre_columns, test_size=0.2):
    train_df, test_df = train_test_split(data, stratify=data['user_id'], test_size=0.2, random_state=42)

    X_train_user = train_df['user'].values.reshape(-1, 1)
    X_train_book = train_df['book'].values.reshape(-1, 1)
    X_train_genre = train_df[genre_columns].values
    y_train = train_df['rating'].values

    X_test_user = test_df['user'].values.reshape(-1, 1)
    X_test_book = test_df['book'].values.reshape(-1, 1)
    X_test_genre = test_df[genre_columns].values
    y_test = test_df['rating'].values

    return train_df, test_df, (X_train_user, X_train_book, X_train_genre, y_train), \
           (X_test_user, X_test_book, X_test_genre, y_test)