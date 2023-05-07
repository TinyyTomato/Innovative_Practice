# encoding: utf-8
import numpy as np
import impact_query_expert_finding.language_models.wrapper
import scipy.sparse
import os
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import pickle
import impact_query_expert_finding.data.io as io
import impact_query_expert_finding.script.document_embedding  as embedded
import impact_query_expert_finding.data.sets as  sets
import heapq
import time
import torch
from impact_query_expert_finding.compress.net import Normalize, forward_pass


# Normalize by setting negative scores to zero and
# dividing by the norm 2 value
def numpy_norm2(in_arr):
    in_arr.clip(min=0)
    norm = np.linalg.norm(in_arr)
    if norm > 0:
        return in_arr / norm
    return in_arr


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
        # print("Model:", parameter['model_name'], "  vote:", self.vote)
        self.parameter = parameter
        self.embedding_PG_index = None
        if parameter['model_name'] is not "tfidf":
            # self.embedding_docs_vectors = np.array(io.load_as_json(parameter['model_dir'], parameter['model_name']))
            self.embedding_docs_vectors = np.array(io.load_as_json(parameter['model_dir'], "trans"))
            # self.embedding_PG_index = embedded.load_PG_index(parameter['model_dir'], parameter['model_name'])
            self.length = len(self.embedding_docs_vectors)
        else:
            doc_rep_dir = os.path.join(self.input_dir, "documents_representations")
            self.language_model = impact_query_expert_finding.language_models.wrapper.LanguageModel(doc_rep_dir,
                                                                                                    type=self.type)
        self.dataset = dataset
        self.k = self.parameter['k']
        # self.embedding_auth = np.array(
        #     io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "F"))

        self.embedding_auth = np.array(
            io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "AEN"))
        # self.encoder = torch.load('/ddisk/lj/DBLP/data/V4/dataset_associations/embedding/triplet.pth')
        # io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "V4_0.8_PAP_15_core_AE"))
        # self.candidates_scores = np.zeros(6012)
        # self.others = [self.k + 100] * (self.length - self.k)
        # self.all = range(0, self.length)

    def predict(self, i, query, leave_one_out=None, k=None):
        candidates_scores = []
        candidates_scores_doc_num = []
        if self.parameter['model_name'] is "tfidf":
            documents_scores = self.language_model.compute_similarity(query)
            documents_sorting_indices = documents_scores.argsort()[::-1]
        else:
            query_vector_emb = self.embedding_docs_vectors[i]
            # documents_scores = np.zeros(self.length)
            if not self.parameter['index']:
                time_start = time.time()
                res_distance, res_index = self.embedding_PG_index.search(np.array([query_vector_emb], dtype='float32'),
                                                                         1000)
                documents_scores = res_distance[0][::-1]
                documents_sorting_indices = res_index[0][::-1]

                print("ALllen=", len(self.embedding_docs_vectors), "query_num=", 1000, "document_rank time:",
                      time.time() - time_start)
                candidates_scores, candidates_scores_doc_num = self.candidates_avg(documents_scores,
                                                                                   documents_sorting_indices, 1)
            else:
                # time_start = time.time()
                documents_scores = np.squeeze(query_vector_emb.dot(self.embedding_docs_vectors.T))
                # documents_scores = vector_matrix(query_vector_emb, self.embedding_docs_vectors)
                documents_sorting_indices = documents_scores.argsort()[::-1]
                # print("ALllen=", len(self.embedding_docs_vectors), "query_num=", 1000, "document_rank time:",
                #       time.time() - time_start)
                candidates_scores, candidates_scores_doc_num = self.candidates_avg(documents_scores,
                                                                                   documents_sorting_indices, 0)

        return candidates_scores, candidates_scores_doc_num

    def predict_with_author_embedding(self, i, query, leave_one_out=None, k=None):
        query_vector_emb = self.embedding_docs_vectors[i]
        author_scores = query_vector_emb.dot(self.embedding_auth.T)
        # return author_scores, None
        author_s = []
        for i in range(0, len(author_scores), int(len(author_scores) / 4232)):
            first = 0
            second = 0
            third = 0
            forth = 0
            five = 0
            for j in range(0, int(len(author_scores) / 4232)):
                if author_scores[i + j] > first:
                    five = forth
                    forth = third
                    third = second
                    second = first
                    first = author_scores[i + j]
                elif author_scores[i + j] > second:
                    five = forth
                    forth = third
                    third = second
                    second = author_scores[i + j]
                elif author_scores[i + j] > third:
                    five = forth
                    forth = third
                    third = author_scores[i + j]
                elif author_scores[i + j] > forth:
                    five = forth
                    forth = author_scores[i + j]
                elif author_scores[i + j] > five:
                    five = author_scores[i + j]
            author_s.append(first + second + third + forth)
        return np.array(author_s), None

    def candidates_avg(self, documents_scores, document_sorting_indices, tag=0):
        A_da = self.dataset.ds.associations
        d_a = A_da.toarray()
        le = len(d_a.T)
        author_scores = [-1.01 for i in range(0, le)]
        author_scores_doc_num = [[] for i in range(-1, le)]
        time_start = time.time()
        if tag == 1:
            for i, doc in enumerate(document_sorting_indices):
                doc_score = documents_scores[i]
                authors_index = np.flatnonzero(d_a[doc])
                for idx in authors_index:
                    author_scores[idx] += doc_score
                    # author_scores_doc_num[idx]+=1
                    author_scores_doc_num[idx].append(doc)
        else:
            for doc_id in document_sorting_indices[:self.parameter['k']]:
                doc_socre = documents_scores[doc_id]
                authors_index = np.flatnonzero(d_a[doc_id])
                for idx in authors_index:
                    author_scores[idx] += doc_socre
                    # author_scores_doc_num[idx]+=1
                    author_scores_doc_num[idx].append(doc_id)

        # np.sort(author_scores)
        # print("len()", len(author_scores), " author:", time.time() - time_start)
        return np.array(author_scores), np.array(author_scores_doc_num)

    def candidates_scores(self, document_sorting_indices):
        A_da = self.dataset.ds.associations
        d_a = A_da.toarray()
        # print(len(A_da.T))
        le = len(d_a.T)
        author_scores = [0 for i in range(0, le)]
        for i, doc in enumerate(document_sorting_indices):
            authors_index = np.flatnonzero(d_a[doc])
            for idx in authors_index:
                # print(a)
                author_scores[idx] += 1 / (i + 1)
        # author_scores.sort()
        return np.array(author_scores)

    def candidates_rank_baseTA(self, document_sorting_indices, k):
        A_da = self.dataset.ds.associations
        d_a = A_da.toarray()
        # print(len(A_da.T))
        le = len(d_a.T)
        author_scores = [0 for i in range(0, le)]

        buffer = []
        dict = {}

        for i, doc in enumerate(document_sorting_indices):
            authors_index = np.flatnonzero(d_a[doc])
            for idx in authors_index:
                if idx not in dict:
                    a_idx = CompareAble(idx=i, lwbound=1 / (i + 1), upbound=7.89873487302022)
                else:
                    dict[idx].lwbound += 1 / (i + 1)
                author_scores[idx] += 1 / (i + 1)
        return np.array(author_scores)


def compute_times(self):
    da = self.dataset.ds.associations


def run_time(self, i, query, leave_one_out=None, k=None):
    query_vector_emb = self.embedding_docs_vectors[i]
    if self.parameter['index']:
        res_distance, res_index = self.embedding_PG_index.search(np.array([query_vector_emb], dtype='float32'),
                                                                 k)
        # print("k", k)
    else:
        # print("k", "no")
        documents_scores = np.squeeze(query_vector_emb.dot(self.embedding_docs_vectors.T))
        documents_sorting_indices = documents_scores.argsort()[::-1]


def vector_matrix(arr, brr):
    return arr.dot(brr.T) / (np.sqrt(np.sum(arr * arr)) * np.sqrt(np.sum(brr * brr, axis=1)))

# 对所有文章[2, 22, 91, 213, 1,...,12]  每篇文章的排名. k 后面的排名统统为len() - 1.
# document_ranks = documents_sorting_indices.argsort() + 1  # 当前方法的document_rank index 序号.
# candidates_scores = np.ravel(
#     self.dataset.ds.associations.T.dot(scipy.sparse.diags(documents_scores, 0)).T.sum(
#         axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b
# return candidates_scores
