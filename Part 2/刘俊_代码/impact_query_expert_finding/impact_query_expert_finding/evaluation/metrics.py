import numpy as np
import sklearn.metrics

"""
'Draw' distribution of precision and recall metrics given scoring/ranking threshold i.e.
best top x (= rank or score) get predicted as positive, the others as negatives.
"""


def get_precision_recall_curve(y_true, y_score):
    return sklearn.metrics.precision_recall_curve(y_true, y_score)


"""
Precision given two predictions like [0,1,0,0,1,0]
"""


def get_precision(y_true, y_score):
    inter_given_true_positives = np.logical_and(y_score, y_true).sum()
    return inter_given_true_positives / y_score.sum()


"""
Recall ...
"""


def get_recall(y_true, y_score):
    inter_given_true_positives = np.logical_and(y_score, y_true).sum()
    return inter_given_true_positives / y_true.sum()


"""
Precision at rank k, where the first k are assigned to positives
"""


def get_precision_at_k(d,y_true, y_score, k):
    if k > len(y_score):
        k = len(y_score)

    top_k_mask = y_score.argsort()[::-1][0:k] # 前k个专家的序号.
    # print("query=",d, "len=", len(y_score)," expert:", top_k_mask)
    return y_true[top_k_mask].sum() / k


"""

"""


# def get_ms_at_k(y_true, y_score, ms_score, k):
#     # y_score . y_score 表示的是 所有expert的分数.
#
#     top_k_mask = y_score.argsort()[::-1][0:k]
#     return ms_score[top_k_mask].sum() / k;


def get_ms_at_k(author_scored_num, y_score, k):
    """
    :type y_score: object
    """
    # if author_scored_num == None:
    #     return 3
    # return 1
    #
    # print(scores)
    # document_num = author_scored_num.sum()

    ans = 0
    author_idx = y_score.argsort()[::-1]  # 作者序号, 按照排名
    for a_idx in author_idx[0:20]:
        ans += (y_score[a_idx] / len(author_scored_num[a_idx]))
    return ans / 20


def get_ads(author_scored_doc, y_score, k, query_id, embeddings):
    """
    :type y_score: object
    """
    if embeddings is None:
        return get_ms_at_k(author_scored_doc, y_score, k)
    author_idx = y_score.argsort()[::-1]  # 作者序号, 按照排名
    y_score = vector_matrix(embeddings[query_id], embeddings)
    author_scored = [0 for i in range(0, len(y_score))]
    ans = 0
    for a_idx in author_idx[0:20]:
        doc_idx = author_scored_doc[a_idx]
        for doc in doc_idx:
            author_scored[a_idx] += y_score[doc]
        if len(doc_idx) is not 0:
            author_scored[a_idx] /= len(doc_idx)
        ans += author_scored[a_idx]
    return ans / 20


"""
Recall at rank k, where the first k are assigned to positives (then always equal to 1)
"""


def get_recall_at_k(y_true, y_score, k):
    # according the score and, sort
    top_k_mask = y_score.argsort()[::-1][0:k]
    return y_true[top_k_mask].sum() / y_true[top_k_mask].sum()


"""
Average precision
"""


def get_average_precision(y_true, y_score):
    return sklearn.metrics.average_precision_score(y_true, y_score)


"""
Reciprocal rank
"""


def get_reciprocal_rank(y_true, y_score):
    sorting_index = y_score.argsort()[::-1]
    for k, i in enumerate(sorting_index):
        if y_true[i] == 1:
            return k + 1
    return len(y_true)


"""
ROC curve
"""


def get_roc_curve(y_true, y_score):
    return sklearn.metrics.roc_curve(y_true, y_score)


"""
ROC AUC score
"""


def get_roc_auc_score(y_true, y_score):
    return sklearn.metrics.roc_auc_score(y_true, y_score)


"""Metrics to compute the model performance."""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer


def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.

    Returns
    -------
    score : float

    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)


# NDCG Scorer function
ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)


def vector_matrix(arr, brr):
    return arr.dot(brr.T) / (np.sqrt(np.sum(arr * arr)) * np.sqrt(np.sum(brr * brr, axis=1)))
