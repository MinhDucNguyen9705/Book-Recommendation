import tensorflow as tf
from tensorflow import keras
import argparse
from NeuMF.model import GMF, MLP, NeuMF
from NeuMF.utils import load_data, preprocess_data, split_data
# from model import GMF, MLP, NeuMF
# from utils import load_data, preprocess_data, split_data
import numpy as np

def parse_options():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train neural network model')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001,
                        help='Learning rate for training (default: 0.001)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--batch_size', '-bs', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--latent_dim', '-ld', type=int, default=100,
                        help='Latent dimension for embeddings (default: 100)')
    parser.add_argument('--layers', '-l', type=int, nargs='+', default=[128,64,64],
                        help='List of layer sizes for MLP (default: [128,64,64])')
    parser.add_argument('--regularization', '-r', type=float, default=0.001,
                        help='Regularization parameter (default: 0.001)')
    
    
    # Parse arguments
    args = parser.parse_args()

    return args

def train_model(model, train_data, val_data, epochs=10, batch_size=256, learning_rate=0.001, save_path='../weights/NeuMF.weights.h5'):
    
    X_train_user, X_train_book, X_train_genre, y_train = train_data
    X_val_user, X_val_book, X_val_genre, y_val = val_data

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mse', 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                                    filepath=save_path, 
                                                    monitor='val_root_mean_squared_error', 
                                                    save_best_only=True,   
                                                    save_weights_only=True,
                                                    mode='min',                 
                                                    verbose=1            
                                                )
    
    # Train the model
    history = model.fit([X_train_user, X_train_book, X_train_genre], y_train,
                        validation_data=([X_val_user, X_val_book, X_val_genre], y_val), 
                        epochs=epochs, 
                        batch_size=batch_size,
                        callbacks=[checkpoint])
    
    return history

class NonBiasedModelTrainer(tf.keras.Model):
    def __init__(self, num_users, num_items, num_genres, latent_dim, layers_, regs, **kwargs):
        super(NonBiasedModelTrainer, self).__init__(**kwargs)
        self.model = NeuMF(num_users, num_items, num_genres, latent_dim, layers_, regs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rmse_metrics = tf.keras.metrics.RootMeanSquaredError(name='rmse')

    @tf.function
    def train_step(self, data):
        (user_ids, book_ids, book_genres, pop_weight), y_true = data
        book_ids = tf.squeeze(book_ids, axis=-1)
        book_ids = tf.cast(book_ids, tf.int64)
        y_true = tf.cast(y_true, tf.float32)
        with tf.GradientTape() as tape:
            y_pred = self.model([user_ids, book_ids, book_genres])
            mse = tf.reduce_mean(tf.square(y_pred - y_true))
            loss = tf.reduce_mean(mse * (1 - pop_weight))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.rmse_metrics.update_state(y_true, y_pred)
        return {
                'loss':self.loss_tracker.result(),
                'rmse':self.rmse_metrics.result()
               }
    
    @tf.function
    def test_step(self, data):
        (user_ids, book_ids, book_genres, pop_weight), y_true = data
        y_true = tf.cast(y_true, tf.float32)
        book_ids = tf.squeeze(book_ids, axis=-1)
        book_ids = tf.cast(book_ids, tf.int64)

        with tf.GradientTape() as tape:
            y_pred = self.model([user_ids, book_ids, book_genres])
            mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)
            loss = tf.reduce_mean(mse * (1 - pop_weight))
        self.loss_tracker.update_state(loss)
        self.rmse_metrics.update_state(y_true, y_pred)
        return {
                'val_loss':self.loss_tracker.result(),
                'val_rmse':self.rmse_metrics.result()
               }

    @property
    def metrics(self):
        return [self.loss_tracker, self.rmse_metrics]

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.rmse_metrics.reset_state()

if __name__ == "__main__":

    args = parse_options()
    
    data, books = load_data('../data/interaction.csv', '../data/final_books.csv')
    data, books, user_encoder, book_encoder, genre_encoder, genre_columns = preprocess_data(data, books)

    train_df, val_df, train_data, val_data = split_data(data, genre_columns, test_size=0.2)
    # train_df, val_df, train_data, val_data = split_data(train_df, genre_columns, test_size=0.1875)

    NUM_USERS = len(data['user_id'].unique())
    NUM_ITEMS = len(data['book_id'].unique())
    NUM_GENRES = len(genre_columns)
    LATENT_DIM = args.latent_dim
    LAYERS = args.layers
    REGULARIZATION = [args.regularization for _ in range (5)]

    # Create the model 
    model = NeuMF(num_users=NUM_USERS, num_items=NUM_ITEMS, num_genres=NUM_GENRES, latent_dim=LATENT_DIM, layers_=LAYERS, regs=REGULARIZATION)
    
    # Train the model with command line arguments
    train_model(model, train_data, val_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate)