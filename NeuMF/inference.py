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

        avg_ratings = self.data.groupby('book')['rating'].mean().to_dict()
        max_avg_ratings = max(avg_ratings.values())
        self.normalized_avg_ratings = {k: v / max_avg_ratings for k, v in avg_ratings.items()}

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

    def update_user(self, new_user, data, learning_rate=0.0001, epochs=10, batch_size=32, save_path='../weights/new_NeuMF.weights.h5'):
        
        user_ids = data['user_id'].unique()
        data = data.merge(self.books, on='book_id')
        data['book'] = data['book_id'].map(lambda x: self.book_encoder.transform([x])[0]).astype(int)
        if new_user:
            num_new_users = len(data['user_id'].unique())
            for uid, new_uid in zip(user_ids, range(self.NUM_USERS, self.NUM_USERS + num_new_users)):
                self.user_map[uid] = new_uid
            model = NeuMF(num_users=self.NUM_USERS+num_new_users, num_items=self.NUM_ITEMS, num_genres=self.NUM_GENRES, latent_dim=self.LATENT_DIM, layers_=self.LAYERS, regs=self.REGS)
        else:
            num_new_users = 0
            model = NeuMF(num_users=self.NUM_USERS, num_items=self.NUM_ITEMS, num_genres=self.NUM_GENRES, latent_dim=self.LATENT_DIM, layers_=self.LAYERS, regs=self.REGS)
        data['user'] = data['user_id'].map(self.user_map)
        _, _, (X_train_user, X_train_book, X_train_genre, y_train), \
        (X_test_user, X_test_book, X_test_genre, y_test) = split_data(data, self.genre_columns, test_size=0.2)
        train_pop_tensor = tf.convert_to_tensor([
            self.normalized_avg_ratings.get(X_train_book[i][0], 0.0) for i in range (len(X_train_book))
        ], dtype=tf.float32)
        test_pop_tensor = tf.convert_to_tensor([
            self.normalized_avg_ratings.get(X_test_book[i][0], 0.0) for i in range (len(X_test_book))
        ], dtype=tf.float32)
        model = self.copy_weights(self.model, model, self.NUM_USERS, self.NUM_ITEMS, self.NUM_GENRES)
        # train_model(model, train_data, val_data,
        #         epochs=epochs,
        #         batch_size=batch_size,
        #         learning_rate=learning_rate,
        #         save_path=save_path)
        trainable_user_indices = list(data['user'].unique())
        wrapped_model = PartialTrainModel(model, trainable_user_indices, learning_rate)
        wrapped_model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        checkpoint = BaseModelCheckpoint(
                                            base_model=wrapped_model.base_model,
                                            filepath=save_path, 
                                            monitor='val_rmse', 
                                            save_best_only=True,   
                                            mode='min',                            
                                        )
        wrapped_model.fit(
                            [X_train_user, X_train_book, X_train_genre, train_pop_tensor],
                            y_train, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=([X_test_user, X_test_book, X_test_genre, test_pop_tensor], y_test),
                            shuffle=True, 
                            verbose=1, 
                            callbacks=[checkpoint]
                        )
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

# class PartialTrainModel(tf.keras.Model):
#     def __init__(self, base_model, new_user_indices, learning_rate=0.001):
#         super().__init__()
#         self.base_model = base_model
#         self.new_user_indices = tf.convert_to_tensor(new_user_indices, dtype=tf.int32)
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#         # Freeze all layers
#         for layer in self.base_model.layers:
#             layer.trainable = False

#         # Unfreeze only user embedding layers
#         self.gmf_user_emb = self.base_model.get_layer('GMF').user_embedding
#         self.mlp_user_emb = self.base_model.get_layer('MLP').user_embedding
#         self.gmf_user_emb.trainable = True
#         self.mlp_user_emb.trainable = True

#         self.loss_tracker = tf.keras.metrics.Mean(name="loss")
#         self.rmse_metrics = tf.keras.metrics.RootMeanSquaredError(name='rmse')

#     def train_step(self, data):
#         x, y = data
#         pop_weight = x[-1]
#         y = tf.cast(y, tf.float32)
#         user_ids = tf.squeeze(x[0], axis=-1)  # assuming shape (batch_size, 1)

#         with tf.GradientTape() as tape:
#             y_pred = self.base_model(x[:-1], training=True)
#             mse = tf.reduce_mean(tf.square(y_pred - y))
#             loss = tf.reduce_mean(mse * (1 - pop_weight))

#         grads = tape.gradient(loss, self.base_model.trainable_weights)

#         # Helper to mask embedding gradients by row
#         def mask_embedding_gradients(grad, emb_var):
#             mask = tf.reduce_any(
#                 tf.equal(tf.range(tf.shape(emb_var)[0])[:, None], self.new_user_indices[None, :]),
#                 axis=1
#             )  # shape: (vocab_size,)
#             mask = tf.cast(mask, tf.float32)
#             mask = tf.expand_dims(mask, axis=-1)  # shape: (vocab_size, 1)
#             return grad * mask

#         masked_grads = []
#         for g, v in zip(grads, self.base_model.trainable_weights):
#             if v is self.gmf_user_emb.embeddings or v is self.mlp_user_emb.embeddings:
#                 g = mask_embedding_gradients(g, v)
#             masked_grads.append(g)

#         self.optimizer.apply_gradients(zip(masked_grads, self.base_model.trainable_weights))
#         self.loss_tracker.update_state(loss)
#         self.rmse_metrics.update_state(y, y_pred)
#         return {
#                 'loss':self.loss_tracker.result(),
#                 'rmse':self.rmse_metrics.result()
#         }
    
#     @tf.function
#     def test_step(self, data):
#         (user_ids, book_ids, book_genres, pop_weight), y_true = data
#         y_true = tf.cast(y_true, tf.float32)
#         book_ids = tf.squeeze(book_ids, axis=-1)
#         book_ids = tf.cast(book_ids, tf.int64)

#         with tf.GradientTape() as tape:
#             y_pred = self.base_model([user_ids, book_ids, book_genres])
#             mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)
#             loss = tf.reduce_mean(mse * (1 - pop_weight))
#         self.loss_tracker.update_state(loss)
#         self.rmse_metrics.update_state(y_true, y_pred)
#         return {
#                 'loss':self.loss_tracker.result(),
#                 'rmse':self.rmse_metrics.result()
#                }

#     @property
#     def metrics(self):
#         return [self.loss_tracker, self.rmse_metrics]

#     def reset_metrics(self):
#         self.loss_tracker.reset_state()
#         self.rmse_metrics.reset_state()

#     def call(self, data, training=False):
#         return self.base_model(data, training=training)
class PartialTrainModel(tf.keras.Model):
    def __init__(self, base_model, trainable_user_indices, learning_rate=0.001):
        super().__init__()
        self.base_model = base_model
        self.trainable_user_indices = tf.convert_to_tensor(trainable_user_indices, dtype=tf.int32)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Freeze all layers in the base model
        for layer in self.base_model.layers:
            layer.trainable = False

        # Unfreeze the user embeddings
        self.gmf_user_emb = self.base_model.get_layer('GMF').user_embedding
        self.mlp_user_emb = self.base_model.get_layer('MLP').user_embedding
        self.gmf_user_emb.trainable = True
        self.mlp_user_emb.trainable = True

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rmse_metrics = tf.keras.metrics.RootMeanSquaredError(name='rmse')

    def mask_embedding_gradients(self, grad, emb_var):
        """Zero-out gradients for all rows except those in trainable_user_indices."""
        vocab_size = tf.shape(emb_var)[0]
        mask = tf.reduce_any(
            tf.equal(tf.range(vocab_size)[:, None], self.trainable_user_indices[None, :]),
            axis=1
        )
        mask = tf.cast(mask, tf.float32)
        return grad * tf.expand_dims(mask, -1)

    def train_step(self, data):
        x, y = data
        pop_weight = x[-1]
        y = tf.cast(y, tf.float32)

        with tf.GradientTape() as tape:
            y_pred = self.base_model(x[:-1], training=True)
            mse = tf.reduce_mean(tf.square(y_pred - y))
            loss = tf.reduce_mean(mse * (1 - pop_weight))

        # Compute gradients only for the user embedding variables
        user_emb_vars = [self.gmf_user_emb.embeddings, self.mlp_user_emb.embeddings]
        grads = tape.gradient(loss, user_emb_vars)

        # Mask gradients
        masked_grads = [
            self.mask_embedding_gradients(g, v)
            for g, v in zip(grads, user_emb_vars)
        ]

        # Apply gradients only to those embedding variables
        self.optimizer.apply_gradients(zip(masked_grads, user_emb_vars))

        self.loss_tracker.update_state(loss)
        self.rmse_metrics.update_state(y, y_pred)
        return {'loss': self.loss_tracker.result(), 'rmse': self.rmse_metrics.result()}

    @tf.function
    def test_step(self, data):
        (user_ids, book_ids, book_genres, pop_weight), y_true = data
        y_true = tf.cast(y_true, tf.float32)
        y_pred = self.base_model([user_ids, book_ids, book_genres])
        mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)
        loss = tf.reduce_mean(mse * (1 - pop_weight))
        self.loss_tracker.update_state(loss)
        self.rmse_metrics.update_state(y_true, y_pred)
        return {'loss': self.loss_tracker.result(), 'rmse': self.rmse_metrics.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.rmse_metrics]

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.rmse_metrics.reset_state()

    def call(self, data, training=False):
        return self.base_model(data, training=training)

class BaseModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, base_model, filepath, save_best_only=False, monitor='val_loss', mode='min'):
        super().__init__()
        self.base_model = base_model
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best = None

        if self.mode == 'min':
            self.monitor_op = tf.math.less
            self.best = float('inf')
        else:
            self.monitor_op = tf.math.greater
            self.best = float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            print(f"[WARNING] Metric {self.monitor} not found in logs.")
            return

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                print(f"Saving best base_model at epoch {epoch+1} with {self.monitor}: {current:.4f}")
                self.best = current
                self.base_model.save_weights(self.filepath.format(epoch=epoch + 1, **logs))
        else:
            print(f"Saving base_model at epoch {epoch+1}")
            self.base_model.save_weights(self.filepath.format(epoch=epoch + 1, **logs))

    
if __name__ == "__main__":
    weight_path = '../weights/non_biased_NeuMF.weights.h5'
    interaction_file = '../data/interaction.csv'
    book_file = '../data/final_books.csv'

    ncf = NeuralMatrixFactoration(weight_path, interaction_file, book_file)
    # user_id = np.random.choice(ncf.data['user_id'].unique())
    user_id = 75853
    book_id = np.random.choice(ncf.data['book_id'].unique())   
    print(ncf.book_encoder.transform([book_id]))
    print(ncf.user_encoder.classes_)
    # rating = ncf.get_user_ratings(user_id)
    # print(f"Predicted rating for user {user_id} and book {book_id}: {rating}")
    # print(f"Predicted ratings for user {user_id}:")
    # for book_id, rating in rating.items():
    #     print(f"Book ID: {book_id}, Predicted Rating: {rating}")

    top_k_books = ncf.get_top_k_recommendations(user_id, k=20)
    # highest_rated_books = ncf.data.groupby('book_id')['rating'].mean().sort_values(ascending=False).index[:50]
    # most_rated_books = ncf.data.groupby('book_id')['rating'].mean().sort_values(ascending=False).index[:50]
    print(f"Top {len(top_k_books)} recommendations for user {user_id}:")
    i = 0
    for book_id, rating in top_k_books:
        # if book_id not in highest_rated_books and book_id not in most_rated_books:
        print(f"Book ID: {book_id}, Book Title: {ncf.books[ncf.books['book_id'] == book_id]['title'].values[0]}, Genre: {ncf.genre_columns[np.argmax(ncf.books[ncf.books['book_id']==book_id][ncf.genre_columns].values[0])]}, Predicted Rating: {rating}, Number of reviews: {len(ncf.data[ncf.data['book_id']==book_id]['user_id'].values)}")
    # new_data = pd.DataFrame({
    #     'user_id': [442590]*6,
    #     'book_id': [66090, 186243, 81662, 184624, 63258, 165016],
    #     'rating': [4, 3, 2, 4, 3, 3],
    # })
    romantic_books = ncf.books[ncf.books['most_tagged']=='romance']['book_id'].values
    random_user = np.random.choice(ncf.data['user_id'].unique(), size=2, replace=False)
    new_data = pd.DataFrame({
        # 'user_id':[np.random.randint(max(ncf.data['user_id'].unique())+1, max(ncf.data['user_id'].unique())+3) for _ in range (100)],
        'user_id':np.random.choice(random_user,size=100),
        'book_id':np.random.choice(romantic_books,size=100,replace=False),
        'rating':[np.random.randint(4, 5) for _ in range (100)]
    })

    ncf.update_user(False, new_data, learning_rate=0.0001, epochs=10, batch_size=4, save_path='../weights/new_non_biased_NeuMF.weights.h5')

    user_id = np.random.choice(new_data['user_id'].unique())
    top_k_books_new = ncf.get_top_k_recommendations(user_id, k=20)
    print(f"Top {len(top_k_books_new)} recommendations for user {user_id}:")
    for book_id, rating in top_k_books_new:
        print(f"Book ID: {book_id}, Book Title: {ncf.books[ncf.books['book_id'] == book_id]['title'].values[0]}, Genre: {ncf.genre_columns[np.argmax(ncf.books[ncf.books['book_id']==book_id][ncf.genre_columns].values[0])]}, Predicted Rating: {rating}, Number of reviews: {len(ncf.data[ncf.data['book_id']==book_id]['user_id'].values)}")
    # #Collision
    # for book_id, rating in top_k_books_new:
    #     if book_id in top_k_books:
    #         print(f"Book ID: {ncf.books[ncf.books['book_id']==book_id]['title'].values[0]}, Predicted Rating: {rating}")