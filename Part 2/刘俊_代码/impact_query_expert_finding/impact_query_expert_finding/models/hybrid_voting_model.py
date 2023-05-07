import numpy as np
import impact_query_expert_finding.preprocessing.text.dictionary
import impact_query_expert_finding.preprocessing.text.vectorizers
import scipy.sparse
import logging
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

logger = logging.getLogger()

path = "/home/lj/tmp/pycharm_project_463/scripts/continue/output"
path2 = "/home/lj/tmp/pycharm_project_463/tests/output"


class Model:

    def __init__(self):
        pass

    def fit(self, A_da, A_dd, T, para=None, model_name=None):
        self.A_da = A_da
        self.Para = para
        logger.debug("Building vocab")
        self.vocab = impact_query_expert_finding.preprocessing.text.dictionary.Dictionary(T, min_df=20, max_df_ratio=0.25)
        logger.debug("Building tfidf vectors")
        self.tfidf_docs_vectors = impact_query_expert_finding.preprocessing.text.vectorizers.get_tfidf_dictionary(self.vocab)

        bert_model = SentenceTransformer(path + model_name);
        bert_model._first_module().max_seq_length = 500
        self.embedding_docs_vectors = normalize(bert_model.encode(T), norm='l2', axis=1)

    def predict(self, d, mask=None):
        query_vector_idf = self.tfidf_docs_vectors[d]
        query_vector_emb = self.embedding_docs_vectors[d]
        # 87/100
        # documents_scores = 0.5 * np.squeeze(query_vector_idf.dot(self.tfidf_docs_vectors.T).A) + np.squeeze( query_vector_emb.dot(self.embedding_docs_vectors.T)) # 对应位置和相加.

        # documents_sorting_indices = documents_scores.argsort()[::-1]
        # document_ranks = documents_sorting_indices.argsort() + 1

        embedding_scores = np.squeeze(query_vector_emb.dot(self.embedding_docs_vectors.T))
        tfidf_scores = np.squeeze(query_vector_idf.dot(self.tfidf_docs_vectors.T).A)

        x = self.Para["i"]
        y = self.Para["j"]
        z = self.Para["k"]
        # 两种 方式的排序结果.
        """
        documents_sorting_indices_embedding = embedding_scores.argsort()[::-1]
        documents_sorting_indices_tfidf = tfidf_scores.argsort()[::-1]

        for i in range(len(documents_sorting_indices_embedding)):
            embedding_scores[documents_sorting_indices_embedding[i]] *= (1 / (x + i + 1))

        for i in range(len(documents_sorting_indices_tfidf)):
            tfidf_scores[documents_sorting_indices_tfidf[i]] *= (1 / (y + i + 1))
        """

        documents_scores = z * embedding_scores + (1 - z) * tfidf_scores
        #
        documents_sorting_indices = documents_scores.argsort()[::-1]

        document_ranks = documents_sorting_indices.argsort() + 1

        # Sort scores and get ranks
        # candidates_scores = np.ravel(
        #     self.A_da.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
        #         axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b
        candidates_scores = np.ravel(
            self.A_da.T.dot(scipy.sparse.diags(documents_scores + tfidf_scores, 0)).T.sum(
                axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b
        if mask is not None:
            candidates_scores = candidates_scores[mask]
        return candidates_scores
