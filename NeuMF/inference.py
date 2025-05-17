import pandas as pd
from model import NeuMF, GMF, MLP
import argparse
import tensorflow as tf
from tensorflow import keras
from utils import load_data, preprocess_data, split_data
import numpy as np
import os 

class NeuralMatrixFactoration():
    def __init__(self, weight_path, interaction_file, book_file):
        self.data, self.books = load_data(interaction_file, book_file)
        self.data, self.books, self.user_encoder, self.book_encoder, self.genre_columns = preprocess_data(self.data, self.books)

        NUM_USERS = len(self.data['user_id'].unique())
        NUM_ITEMS = len(self.data['book_id'].unique())
        NUM_GENRES = len(self.genre_columns)
        LATENT_DIM = 100
        LAYERS = [128, 64, 64]
        REGS = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]

        self.model = NeuMF(num_users=NUM_USERS, num_items=NUM_ITEMS, num_genres=NUM_GENRES, latent_dim=LATENT_DIM, layers_=LAYERS, regs=REGS)
        self.model.load_weights(weight_path)

    def get_user_book_pair_rating(self, user_id, book_id):

        user_input = np.array([[user_id]])
        book_input = np.array([[book_id]])

        user_input = self.user_encoder.transform(user_input)
        book_input = self.book_encoder.transform(book_input)
        
        genre = self.books[self.books['book_id'] == book_id][self.genre_columns].values
        genre_input = np.array([genre]).reshape((-1,10))

        print(user_input.shape)
        print(book_input.shape)
        print(genre_input.shape)

        rating = self.model.predict([user_input, book_input, genre_input], verbose=0)

        return rating[0][0]

    def get_user_ratings(self, user_id):

        user_input = np.array([[user_id]])
        user_input = self.user_encoder.transform(user_input)

        values = self.books[['book_id'] + list(self.genre_columns)].values
        book_ids = values[:, 0]
        genre = values[:, 1:]
        book_input = self.book_encoder.transform(book_ids)
        book_input = np.expand_dims(np.array(book_input), axis=1)
        genre_input = np.array([genre]).reshape((-1,10))
        user_input = np.full((book_input.shape[0], 1), user_input)
        predictions = self.model.predict([user_input, book_input, genre_input], verbose=0)
        book_ids_flat = book_ids.flatten()
        predictions_flat = predictions.flatten()
        
        book_pred_dict = dict(zip(book_ids_flat, predictions_flat))

        return book_pred_dict

    def get_top_k_recommendations(self, user_id, k=10):
        book_pred_dict = self.get_user_ratings(user_id)
        sorted_books = sorted(book_pred_dict.items(), key=lambda x: x[1], reverse=True)
        top_k_books = sorted_books[:k]
        return top_k_books
    
if __name__ == "__main__":
    weight_path = 'NeuMF.weights.h5'
    interaction_file = 'interaction.csv'
    book_file = 'final_books.csv'

    df = pd.read_csv(interaction_file)
    print(df.head())

    ncf = NeuralMatrixFactoration(weight_path, interaction_file, book_file)
    user_id = np.random.choice(ncf.data['user_id'].unique())
    book_id = np.random.choice(ncf.data['book_id'].unique())   
    # rating = ncf.get_user_ratings(user_id)
    # print(f"Predicted rating for user {user_id} and book {book_id}: {rating}")
    # print(f"Predicted ratings for user {user_id}:")
    # for book_id, rating in rating.items():
    #     print(f"Book ID: {book_id}, Predicted Rating: {rating}")

    top_k_books = ncf.get_top_k_recommendations(user_id, k=10)
    print(f"Top {len(top_k_books)} recommendations for user {user_id}:")
    for book_id, rating in top_k_books:
        print(f"Book ID: {book_id}, Predicted Rating: {rating}")