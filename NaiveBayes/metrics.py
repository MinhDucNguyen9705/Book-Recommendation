import numpy as np
def dcg_k(r, k):
    """ Discounted Cumulative Gain (DGC)  
    Args:
        r: True Ratings in Predicted Rank Order (1st element is top recommendation)
        k: Number of results to consider
    Returns:
        DCG
    """
  
    r = np.asfarray(r)[:k]
    return np.sum(2**r / np.log2(np.arange(2, r.size + 2)))

def ndcg_k(r, k):
    """Normalized Discounted Cumulative Gain (NDCG)
    Args:
        r: True Ratings in Predicted Rank Order (1st element is top recommendation)
        k: Number of results to consider
    Returns:
        NDCG
    """
    dcg_max = dcg_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_k(r, k) / dcg_max

def mean_ndcg(rs):
    """Mean NDCG for all users
    Args:
        rs: Iterator / For each user: True Ratings in Predicted Rank Order
    Returns:
        Mean NDCG
    """
    return np.mean([ndcg_k(r, len(r)) for r in rs])

def rmse(y,h):
    """RMSE
    Args:
        y: real y
        h: predicted y
    Returns:
        RMSE
    """
    a = y-h

    return np.sqrt(sum(a**2)/len(a))