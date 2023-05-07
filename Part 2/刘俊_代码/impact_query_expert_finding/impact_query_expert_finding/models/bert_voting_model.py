import scipy.sparse
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import scipy

logger = logging.getLogger()
path = "/home/lj/tmp/pycharm_project_463/scripts/continue/output"
class Model:

    def __init__(self):
        pass

    def fit(self, A_da, A_dd, T , model_path):
        self.A_da = A_da

        # bert_model = SentenceTransformer(path + "/sci_bert_nil")
        # bert_model = SentenceTransformer(path + "/sci_bert_nil_sts")
        # bert_model = SentenceTransformer(path + "/doc_doc_sci_bert_triples")
        # bert_model = SentenceTransformer(path + "/doc_doc_sci_bert_nil_sts_triples")
        # bert_model = SentenceTransformer(path + "/doc_doc_sci_bert_triples_nil_sts")
        # bert_model = SentenceTransformer(path + "/doc_doc_sci_bert_triples_lexical")
        bert_model = SentenceTransformer(path + model_path)
        bert_model._first_module().max_seq_length = 500
        self.docs_vectors = bert_model.encode(T)

    def predict(self, d, mask=None):
        query_vector = self.docs_vectors[d]
        documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T))

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
