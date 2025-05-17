import tensorflow as tf
from tensorflow import keras
from model import GMF, MLP, NeuMF
import argparse
from utils import load_data, preprocess_data, split_data
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

def train_model(model, train_data, val_data, epochs=10, batch_size=256, learning_rate=0.001):
    
    X_train_user, X_train_book, X_train_genre, y_train = train_data
    X_val_user, X_val_book, X_val_genre, y_val = val_data

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    # Train the model
    history = model.fit([X_train_user, X_train_book, X_train_genre], y_train,
                        validation_data=([X_val_user, X_val_book, X_val_genre], y_val), 
                        epochs=epochs, 
                        batch_size=batch_size)
    
    return history

if __name__ == "__main__":

    args = parse_options()
    
    data = load_data('interaction.csv', 'final_books.csv')
    data, genre_columns = preprocess_data(data)

    train_data, val_data, test_data = split_data(data, genre_columns)

    NUM_USERS = len(data['user_id'].unique())
    NUM_ITEMS = len(data['book_id'].unique())
    NUM_GENRES = len(genre_columns)
    LATENT_DIM = args.latent_dim
    LAYERS = args.layers
    REGULARIZATION = [args.regularization for _ in range (5)]

    # Create the model 
    model = NeuMF(num_users=NUM_USERS, num_items=NUM_ITEMS, num_genres=NUM_GENRES, layers_=LAYERS, regs=REGULARIZATION)
    
    # Train the model with command line arguments
    train_model(model, train_data, val_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate)