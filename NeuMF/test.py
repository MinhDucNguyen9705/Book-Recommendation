from model import NeuMF, GMF, MLP
import argparse
import tensorflow as tf
from tensorflow import keras
from utils import load_data, preprocess_data, split_data
import numpy as np
import os 

def load_model(file_path):
    model = keras.models.load_model(file_path)
    return model

def get_user_rating(model, user_id, book_id, genre):
    user_input = np.array([[user_id]])
    book_input = np.array([[book_id]])
    rating = model.predict([user_input, book_input])
    return rating[0][0]