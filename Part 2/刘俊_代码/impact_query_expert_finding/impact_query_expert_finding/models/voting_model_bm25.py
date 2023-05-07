import scipy.sparse
from rank_bm25 import BM25Okapi
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import scipy
logger = logging.getLogger()

class Model:

    def __init__(self):
        pass

    def fit(self, A_da, A_dd, T):
        self.A_da = A_da
        self.T = T
        maxLen = 0;
        for d in T:
            maxLen = max(maxLen, len(d.split()))
        print(maxLen)
        # bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        # "scripts/continue/output/training_dblpbert-base-uncased-2020-09-14_19-25-37"
        # bert_model = SentenceTransformer('preprocessing/train_model/output/training_stsbenchmark_avg_word_embeddings-2020-08-20_14-02-24')
        bert_model = SentenceTransformer("/home/lj/tmp/pycharm_project_463/tests/output/training_stsbenchmark_continue_training-sci_bert_nil-2020-09-22_08-50-29")
        bert_model._first_module().max_seq_length = 500
        # bert_model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        # bert_model = SentenceTransformer('bert-base-nli-stsb-wkpooling')
        # bert_model = SentenceTransformer("roberta-large-nli-stsb-mean-tokens")
        tokenized_corpus = [doc.split(" ") for doc in T]
        self.bm25 = BM25Plus(tokenized_corpus)

        self.docs_vectors = bert_model.encode(T)


    def predict(self, d, mask=None):
        # query_vector = self.docs_vectors[d]
        tokenized_query = self.T[d].split(" ")
        documents_scores = self.bm25.get_scores(tokenized_query)
        # documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T))
        # documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T).A)
        documents_sorting_indices = documents_scores.argsort()[::-1]
        document_ranks = documents_sorting_indices.argsort() + 1

        # Sort scores and get ranks // Rank sore = 1 / docuement_ranks
        candidates_scores = np.ravel(
            self.A_da.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
                axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b

        if mask is not None:
            candidates_scores = candidates_scores[mask]
        return candidates_scores
