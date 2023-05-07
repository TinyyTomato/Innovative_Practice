import numpy as np
import impact_query_expert_finding.language_models.wrapper
import scipy.sparse
import os
import numpy as np
import scipy.sparse
import logging
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
# import faiss
import pickle


# Normalize by setting negative scores to zero and
# dividing by the norm 2 value
def numpy_norm2(in_arr):
    in_arr.clip(min=0)
    norm = np.linalg.norm(in_arr)
    if norm > 0:
        return in_arr / norm
    return in_arr


# dir = "/home/lj/tmp/pycharm_project_463/scripts/continue/output"

# dir = "/ddisk/lj/V2"

# describes = []
# with open("/home/lj/project/impact_query_expert_finding/script/output/data/topics_desc_bak.txt", "r") as f:
#     data = f.readlines()
#     for line in data:
        # describes.append(line)


class VotingModel:

    def __init__(self, config, type="tfidf", vote="rr", **kargs):
        self.type = kargs["language_model"]
        if "vote_technique" in kargs:
            self.vote = kargs["vote_technique"]
        else:
            self.vote = "panoptic"
        self.config = config
        self.dataset = None
        self.parameter = None
        self.embedding_docs_vectors = None
        self.language_model = None
        self.input_dir = kargs["input_dir"]

    def fit(self, x, Y, dataset=None, path=None, parameter=None, mask=None):
        self.parameter = parameter
        doc_rep_dir = os.path.join(self.input_dir, "documents_representations")
        self.language_model = impact_query_expert_finding \
            .language_models \
            .wrapper.LanguageModel(doc_rep_dir, type=self.type)

        self.dataset = dataset
        if parameter['model_name'] is not "tfidf":
            with open(parameter['model_dir'] + parameter['model_name'], "rb") as f:
                self.embedding_docs_vectors = normalize(pickle.load(f))

    def predict(self, i, query, leave_one_out=None, topic_as_queries=None):
        if self.parameter['model_name'] is "tfidf":
            tfidf_scores = self.language_model.compute_similarity(query)
            documents_scores = tfidf_scores
        else:
            query_vector_emb = self.embedding_docs_vectors[i]
            documents_scores = np.squeeze(query_vector_emb.dot(self.embedding_docs_vectors.T))

        documents_sorting_indices = documents_scores.argsort()[::-1]
        document_ranks = documents_sorting_indices.argsort() + 1
        # Sort scores and get ranks
        if self.parameter['rank_type'] is "vote":
            candidates_scores = np.ravel(
                self.dataset.ds.associations.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
                    axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b
            return candidates_scores
        else :
            candidates_scores = np.ravel(
                self.dataset.ds.associations.T.dot(
                    scipy.sparse.diags(documents_scores + tfidf_scores, 0)).T.sum(
                    axis=0))
            return candidates_scores
    # todo add some other Strategy


#     candidates_scores = self.dataset.ds.associations.T.dot(
#         scipy.sparse.diags(1 / np.log(document_ranks, 0))).T.sum(axis=0)
#     return candidates_scores

'''
'  if topic_as_queries:
            if self.parameter['query_type'] is "topic_describe":
                path = "/home/lj/tmp/pycharm_project_462/scripts/continue/output"
                encoder = SentenceTransformer(path + self.parameter['model_name'])
                query_vector_emb = normalize(encoder.encode([describes[i]]), norm='l1', axis=1)
                tfidf_scores = self.language_model.compute_similarity(describes[i])
            else:
                # topic word_embedding
                path = "/home/lj/tmp/pycharm_project_462/scripts/continue/output"
                encoder = SentenceTransformer(path + self.parameter['model_name'])
                query_vector_emb = normalize(encoder.encode([query]), norm='l1', axis=1)
                tfidf_scores = self.language_model.compute_similarity(query)
                # query_vector_emb = normalize(encoder.encode([query]), norm='l1', axis=1)
        else:
            print("document_query")'''
