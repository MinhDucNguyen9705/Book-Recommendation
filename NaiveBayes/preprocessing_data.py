import pandas as pd
import collections
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text = text.lower()
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    text = text.replace('  ', ' ')
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

def remove_punctuation(text):
    text = text.replace('.', '')
    text = text.replace(',', '')
    text = text.replace('!', '')
    text = text.replace('?', '')
    text = text.replace(':', '')
    text = text.replace(';', '')
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_rare_words(text):
    words = text.split()
    word_counts = collections.Counter(words)
    rare_words = {word for word, count in word_counts.items() if count < 2}
    text = ' '.join(word for word in words if word not in rare_words)
    return text

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def remove_special_characters(text):
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = remove_punctuation(text)
    text = lemmatize_text(text)
    text = remove_punctuation(text)
    text = remove_rare_words(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    return text

def final_preprocess_text(text_data):
    copy_data = text_data.copy()
    for i in range(len(copy_data)):
        copy_data[i] = preprocess_text(copy_data[i])
    return copy_data

def build_vocab(text_data):
    vocab = []
    for i in range(len(text_data)):
        text = text_data[i]
        text = text.split()
        for word in text:
            if word not in vocab:
                vocab.append(word)
    return vocab

def vectorize_text(text, vocab):
    vector = np.zeros(len(vocab))
    text = text.split()
    for word in text:
        if word in vocab:
            index = vocab.index(word)
            vector[index] += 1
    return vector

def preprocess(file_path):
    des = pd.read_csv(file_path + 'final_books.csv')
    raw_users_books = pd.read_csv(file_path + 'interaction.csv')
    des = des.iloc[:,1:]
    des = des.loc[:,['book_id', 'title', 'description']]
    processed = final_preprocess_text(des['description'])
    vocab = build_vocab(processed)
    word_vec = []
    for i in range(len(processed)):
        word_vec.append(vectorize_text(processed[i], vocab))
    des['processed description'] = processed
    des['word_vector'] = word_vec
    merged = pd.merge(raw_users_books, des, on='book_id', how='inner').iloc[:,1:]
    merged.sort_values(by='user_id', ascending=True, inplace=True)
    return merged, vocab
    