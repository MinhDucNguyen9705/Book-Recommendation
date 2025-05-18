import pandas as pd
import numpy as np
def get_words(message):
    words = message
    words = words.split(" ")
    words = [x.lower() for x in words]

    return words

def rmse(y,h): return np.sqrt(np.mean((y-h)**2))
def dcg_k(r,k):
    r = np.asarray(r)[:k]
    return np.sum((2**r-1) / np.log2(np.arange(2, len(r)+2)))
def ndcg_k(r,k):
    idcg = dcg_k(sorted(r, reverse=True), k)
    return dcg_k(r,k)/idcg if idcg>0 else 0.
def mean_ndcg(rs):
    return np.mean([ndcg_k(r,len(r)) for r in rs])