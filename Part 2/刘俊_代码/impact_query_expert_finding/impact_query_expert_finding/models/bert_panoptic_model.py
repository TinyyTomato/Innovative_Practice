import numpy as np
import impact_query_expert_finding.preprocessing.text.dictionary
import impact_query_expert_finding.preprocessing.text.vectorizers
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger()


class Model:

    def __init__(self):
        self.authors_vectors = None
        self.docs_vectors = None

    def fit(self, A_da, A_dd, T):
        logger.debug("Building vocab")
        # self.vocab = impact_query_expert_finding.preprocessing.text.dictionary.Dictionary(T, min_df=5, max_df_ratio=0.25)
        # logger.debug("Building tfidf vectors")
        # self.docs_vectors = impact_query_expert_finding.preprocessing.text.vectorizers.get_tfidf_dictionary(self.vocab)
        bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        # bert_model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        # bert_model = SentenceTransformer('bert-base-nli-stsb-wkpooling')
        self.docs_vectors = bert_model.encode(T)
        authors_metadocs = list()
        for i in range(A_da.shape[1]):
            # authors_metadocs.append(" ".join([T[j] for j in A_da[:, i].nonzero()[0]]))
            docuemnts = [T[j] for j in A_da[:, i].nonzero()[0]]
            docuemnts_vector = bert_model.encode(docuemnts)

            author_vector = np.zeros(docuemnts_vector[0].shape)
            for i, vector in enumerate(docuemnts_vector):
                author_vector += vector
            author_vector /= len(docuemnts_vector)
            # author_vector = np.mean(docuemnts_vector, axis=0)
            # print(author_vector)
            authors_metadocs.append(author_vector)

        # author_document = bert_model.encode(authors_metadocs)
        self.authors_vectors = np.array(authors_metadocs)
        print(self.authors_vectors.shape)
        #self.authors_vectors = impact_query_expert_finding.preprocessing.text.vectorizers.get_tfidf_N(self.vocab, authors_metadocs)

    def predict(self, d, mask=None):
        query_vector = self.docs_vectors[d]
        candidates_scores = np.squeeze(query_vector.dot(self.authors_vectors.T))
        if mask is not None:
            candidates_scores = candidates_scores[mask]
        return candidates_scores
