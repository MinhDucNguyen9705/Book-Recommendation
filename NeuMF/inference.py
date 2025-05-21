import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras
from NeuMF.utils import load_data, preprocess_data, split_data
from NeuMF.train import train_model
from NeuMF.model import NeuMF, GMF, MLP
# from utils import load_data, preprocess_data, split_data
# from train import train_model
# from model import NeuMF, GMF, MLP
import numpy as np
import os 

class NeuralMatrixFactoration():
    def __init__(self, weight_path, interaction_file, book_file):
        self.data, self.books = load_data(interaction_file, book_file)
        self.data, self.books, self.user_encoder, self.book_encoder, self.genre_encoder, self.genre_columns = preprocess_data(self.data, self.books)
        self.user_map = {uid: i for i, uid in enumerate(self.data['user_id'].unique())}
        self.NUM_USERS = len(self.data['user_id'].unique())
        self.NUM_ITEMS = len(self.data['book_id'].unique())
        self.NUM_GENRES = len(self.genre_columns)
        self.LATENT_DIM = 100
        self.LAYERS = [128, 64, 64]
        self.REGS = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]

        self.model = NeuMF(num_users=self.NUM_USERS, num_items=self.NUM_ITEMS, num_genres=self.NUM_GENRES, latent_dim=self.LATENT_DIM, layers_=self.LAYERS, regs=self.REGS)
        self.model.load_weights(weight_path)

    def get_user_book_pair_rating(self, user_id, book_id):

        user_input = np.array([[user_id]])
        book_input = np.array([[book_id]])

        user_input = self.user_encoder.transform(user_input)
        book_input = self.book_encoder.transform(book_input)
        
        genre = self.books[self.books['book_id'] == book_id][self.genre_columns].values
        genre_input = np.array([genre]).reshape((-1,10))

        rating = self.model.predict([user_input, book_input, genre_input], verbose=0)

        return rating[0][0]

    def get_user_ratings(self, user_id):

        if user_id not in self.user_encoder.classes_:
            user_id = self.user_map[user_id]
            user_input = np.array([[user_id]])   
        else:
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

    def update_user(self, data, learning_rate=0.0001, epochs=10, batch_size=32, save_path='../weights/new_NeuMF.weights.h5'):

        num_new_users = len(data['user_id'].unique())
        user_ids = data['user_id'].unique()

        for uid, new_uid in zip(user_ids, range(self.NUM_USERS, self.NUM_USERS + num_new_users)):
            self.user_map[uid] = new_uid
        # self.user_map = {uid: new_uid for uid, new_uid in zip(user_ids, range(self.NUM_USERS, self.NUM_USERS + num_new_users))}
        data = data.merge(self.books, on='book_id')
        # print(data.head())
        # print(data.columns)
        # genre_encoded = self.genre_encoder.transform(data[['most_tagged']])

        # genre_df = pd.DataFrame(genre_encoded, columns=self.genre_columns, index=data.index)

        # data = pd.concat([data.drop(columns=['most_tagged']), genre_df], axis=1)
        data['user'] = data['user_id'].map(self.user_map)
        data['book'] = data['book_id'].map(lambda x: self.book_encoder.transform([x])[0])

        _, _, train_data, val_data = split_data(data, self.genre_columns, test_size=0.2)
        
        model = NeuMF(num_users=self.NUM_USERS+num_new_users, num_items=self.NUM_ITEMS, num_genres=self.NUM_GENRES, latent_dim=self.LATENT_DIM, layers_=self.LAYERS, regs=self.REGS)
        model = self.copy_weights(self.model, model, self.NUM_USERS, self.NUM_ITEMS, self.NUM_GENRES)
        train_model(model, train_data, val_data,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_path=save_path)
        model.load_weights(save_path)
        self.model = model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='mse', 
                        metrics=[tf.keras.metrics.RootMeanSquaredError()])
        self.NUM_USERS += num_new_users
        self.data = pd.concat([self.data, data], ignore_index=True)

    def copy_weights(self, old_model, new_model, num_users_old, num_items_old, num_genres_old):

        old_user_emb = old_model.get_layer('GMF').user_embedding.get_weights()[0]
        new_user_emb = new_model.get_layer('GMF').user_embedding.get_weights()[0]
        new_user_emb[:num_users_old] = old_user_emb
        new_model.get_layer('GMF').user_embedding.set_weights([new_user_emb])

        old_item_emb = old_model.get_layer('GMF').item_embedding.get_weights()[0]
        new_model.get_layer('GMF').item_embedding.set_weights([old_item_emb])

        old_user_emb = old_model.get_layer('MLP').user_embedding.get_weights()[0]
        new_user_emb = new_model.get_layer('MLP').user_embedding.get_weights()[0]
        new_user_emb[:num_users_old] = old_user_emb
        new_model.get_layer('MLP').user_embedding.set_weights([new_user_emb])

        old_item_emb = old_model.get_layer('MLP').item_embedding.get_weights()[0]
        new_model.get_layer('MLP').item_embedding.set_weights([old_item_emb])

        old_genre_emb = old_model.get_layer('MLP').genre_embedding.get_weights()[0]
        new_model.get_layer('MLP').genre_embedding.set_weights([old_genre_emb])

        for i, dense_layer in enumerate(old_model.get_layer('MLP').fcn.layers):
            old_weights = dense_layer.get_weights()
            new_model.get_layer('MLP').fcn.layers[i].set_weights(old_weights)

        old_final_dense = old_model.get_layer('output').get_weights()
        new_model.get_layer('output').set_weights(old_final_dense)

        return new_model

class PartialTrainModel(tf.keras.Model):
    def __init__(self, base_model, new_user_indices):
        super().__init__()
        self.base_model = base_model
        self.new_user_indices = tf.convert_to_tensor(new_user_indices, dtype=tf.int32)

        for layer in self.base_model.layers:
            layer.trainable = False

        self.gmf_user_emb = self.base_model.get_layer('GMF').user_embedding
        self.mlp_user_emb = self.base_model.get_layer('MLP').user_embedding
        self.gmf_user_emb.trainable = True
        self.mlp_user_emb.trainable = True

    def train_step(self, data):
        x, y = data
        user_ids = tf.squeeze(x[0], axis=-1) 

        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        grads = tape.gradient(loss, self.base_model.trainable_weights)

        def mask_embedding_gradients(grad, emb_var):
            mask = tf.reduce_any(
                tf.equal(tf.range(tf.shape(emb_var)[0])[:, None], self.new_user_indices[None, :]),
                axis=1
            ) 
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
            return grad * mask

        masked_grads = []
        for g, v in zip(grads, self.base_model.trainable_weights):
            if v is self.gmf_user_emb.embeddings or v is self.mlp_user_emb.embeddings:
                g = mask_embedding_gradients(g, v)
            masked_grads.append(g)

        self.optimizer.apply_gradients(zip(masked_grads, self.base_model.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, data, training=False):
        return self.base_model(data, training=training)
    
    
if __name__ == "__main__":
    weight_path = '../weights/NeuMF.weights.h5'
    interaction_file = '../data/interaction.csv'
    book_file = '../data/final_books.csv'

    ncf = NeuralMatrixFactoration(weight_path, interaction_file, book_file)
    # user_id = np.random.choice(ncf.data['user_id'].unique())
    user_id = 2142
    book_id = np.random.choice(ncf.data['book_id'].unique())   
    # rating = ncf.get_user_ratings(user_id)
    # print(f"Predicted rating for user {user_id} and book {book_id}: {rating}")
    # print(f"Predicted ratings for user {user_id}:")
    # for book_id, rating in rating.items():
    #     print(f"Book ID: {book_id}, Predicted Rating: {rating}")

    top_k_books = ncf.get_top_k_recommendations(user_id, k=1000)
    # highest_rated_books = ncf.data.groupby('book_id')['rating'].mean().sort_values(ascending=False).index[:50]
    # most_rated_books = ncf.data.groupby('book_id')['rating'].mean().sort_values(ascending=False).index[:50]
    print(f"Top {len(top_k_books)} recommendations for user {user_id}:")
    i = 0
    for book_id, rating in top_k_books:
        # if book_id not in highest_rated_books and book_id not in most_rated_books:
        if ncf.data[ncf.data['book_id']==book_id]['rating'].mean() < 4.2:
            i+=1
            print(f"Book ID: {ncf.books[ncf.books["book_id"]==book_id]['title'].values[0]}, Predicted Rating: {rating}")
            if i == 20:
                break

    # new_data = pd.DataFrame({
    #     'user_id': [442590]*6,
    #     'book_id': [66090, 186243, 81662, 184624, 63258, 165016],
    #     'rating': [4, 3, 2, 4, 3, 3],
    # })

    # ncf.update_user(new_data, learning_rate=0.0001, epochs=10, batch_size=4, save_path='../weights/NeuMF.weights.h5')

    # user_id = np.random.choice(new_data['user_id'].unique())
    # top_k_books_new = ncf.get_top_k_recommendations(user_id, k=1000)
    # highest_rated_books = ncf.data.groupby('book_id')['rating'].mean().sort_values(ascending=False).index[:50]
    # most_rated_books = ncf.data.groupby('book_id')['rating'].mean().sort_values(ascending=False).index[:50]
    # print(f"Top {len(top_k_books_new)} recommendations for user {user_id}:")
    # for book_id, rating in top_k_books_new:
    #     if book_id not in highest_rated_books and book_id not in most_rated_books:
    #         print(f"Book ID: {ncf.books[ncf.books["book_id"]==book_id]['title'].values[0]}, Predicted Rating: {rating}")

    # #Collision
    # for book_id, rating in top_k_books_new:
    #     if book_id in top_k_books:
    #         print(f"Book ID: {ncf.books[ncf.books['book_id']==book_id]['title'].values[0]}, Predicted Rating: {rating}")