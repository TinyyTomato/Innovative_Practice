# encoding: utf-8
import impact_query_expert_finding.main.evaluate_LExR
import impact_query_expert_finding.main.topics_LExR
import os
import time
import numpy as np
import impact_query_expert_finding.data.io as io
import impact_query_expert_finding.script.document_embedding  as embedded
import impact_query_expert_finding.data.sets as  sets
import heapq
import scipy

current_folder = os.path.dirname(os.path.abspath(__file__))



def run(version, model_dir, model_name, work_dir, index, k, dataset_type="dataset_associations"):
    input_dir = os.path.join(work_dir, version, dataset_type)
    output_dir = os.path.join(work_dir, version, dataset_type, "result")
    parameters = {
        'output_dir': output_dir,
        'input_dir': input_dir,
        'algorithm': "vote",
        'data_type': "dataset_associations",
        'model_type': 'embedding',
        'model_name': model_name,
        'model_dir': model_dir,
        'query_type': "document",
        'rank_type': "vote",
        'language_model': "tfidf",
        'vote_technique': 'rr',
        'eta': 0.1,
        'seed': 0,
        'max_queries': 100,
        'dump_dir': output_dir,
        'index': index,
        'k': k,
    }

    begin_time = time.time()
    impact_query_expert_finding.main.evaluate_LExR.run(parameters)
    end_time = time.time()
    print("查询评估所用的平均时间: (topm doucument + topn author + metrics)", (end_time - begin_time), "s")
    # index_spend_time(parameters)


def index_spend_time(parameter):
    embedding_docs_vectors = np.array(io.load_as_json(parameter['model_dir'], parameter['model_name']))
    embedding_PG_index = embedded.load_PG_index(parameter['model_dir'], parameter['model_name'])
    length = len(embedding_docs_vectors)
    print("query nums", length)
    arr = [1000, 500, 100, 50, 5]
    for m in arr:
        time_start = time.time()
        for i in range(length):
            query_vector_emb = embedding_docs_vectors[i]
            res_distance, res_index = embedding_PG_index.search(np.array([query_vector_emb], dtype='float32'), m)
        print(parameter['model_name'], m, " spend time w index ", (time.time() - time_start) * 1000 / length,
              "ms")
    query_vector_emb = embedding_docs_vectors[0]
    documents_scores = np.zeros(length)

    time_start = time.time()
    for i, embed in enumerate(embedding_docs_vectors):
        documents_scores[i] = query_vector_emb.dot(embed)
    documents_sorting_indices = documents_scores.argsort()[::-1]

    print(parameter['model_name'], " spend time w/o index ", (time.time() - time_start) * 1000)
    candidates_scores(documents_sorting_indices, parameter=parameter)


def candidates_scores(document_sorting_indices, parameter):
    dataset = impact_query_expert_finding.data.sets.DataSet("acm")
    dataset_path = parameter["input_dir"]
    dataset.load_LExR(dataset_path)
    A_da = dataset.ds.associations
    d_a = A_da.toarray()
    le = len(d_a.T)
    author_scores = [0 for i in range(0, le)]

    time_start = time.time()
    for i, doc in enumerate(document_sorting_indices):
        authors_index = np.flatnonzero(d_a[doc])
        for idx in authors_index:
            # print(a)
            author_scores[idx] += 1 / (i + 1)
    author_scores.sort()
    print(parameter['model_name'], " spend time w/o ta", (time.time() - time_start) * 1000, "ms")

    document_ranks = document_sorting_indices.argsort() + 1
    time_start = time.time()
    candidates_scores = np.ravel(
        dataset.ds.associations.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
            axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b
    candidates_scores.sort()
    print(parameter['model_name'], " spend time w/o ta matrix", (time.time() - time_start) * 1000, "ms")

    candidates_rank_baseTA(dataset, document_sorting_indices, parameter, 50)

    return np.array(author_scores)


def candidates_scores_TA(self, document_sorting_indices):
    A_da = self.dataset.ds.associations
    d_a = A_da.toarray()
    # print(len(A_da.T))
    le = len(d_a.T)
    author_scores = [0 for i in range(0, le)]
    author_set.add(authors_index)
    for i, doc in enumerate(document_sorting_indices):
        authors_index = np.flatnonzero(d_a[doc])
        for idx in authors_index:
            author_scores[idx] += 1 / (i + 1)
    author_scores.sort()
    return np.array(author_scores)


def candidates_rank_baseTA(dataset, document_sorting_indices, parameter, k):
    A_da = dataset.ds.associations
    d_a = A_da.toarray()
    le = len(d_a.T)
    author_scores = [0 for i in range(0, le)]

    buffer = []
    dict = {}
    nnum = [5, 10, 20, 50, 100]
    for n in nnum:
        time_start = time.time()
        for i, doc in enumerate(document_sorting_indices[0:n * 10]):
            authors_index = np.flatnonzero(d_a[doc])
            for idx in authors_index:
                if idx not in dict:
                    a_idx = CompareAble(idx=i, lwbound=1 / (i + 1), upbound=7.21)
                else:
                    dict[idx].lwbound += 1 / (i + 1)
                author_scores[idx] += 1 / (i + 1)

        print(parameter['model_name'], " spend time w ta", (time.time() - time_start) * 1000, "ms")

    return np.array(author_scores)


class CompareAble:
    def __init__(self, idx, lwbound, upbound):
        self.id = idx
        self.lwbound = lwbound
        self.upbound = upbound

    def __lt__(self, other):
        return self.lwbound < other.lwbound

    def __cmp__(self, other):
        if self.lwbound < other.lwbuound:
            return -1
        elif self.lwbound == other.lwbuound:
            return self.upbound < self.upbound
        else:
            return 1


def TA_time():
    A_da = self.dataset.ds.associations
    d_a = A_da.toarray()
    # print(len(A_da.T))
    le = len(d_a.T)
    author_scores = [0 for i in range(0, le)]
    start_time = time.time()
    for i, doc in enumerate(document_sorting_indices):
        authors_index = np.flatnonzero(d_a[doc])
        for idx in authors_index:
            author_scores[idx] += 1 / (i + 1)
    candidate_score = np.sort(author_scores)
    print(parameter['model_name'], " spend time w ta", time.time() - time_start)
    print(parameter['model_name'], " spend time w/o ta", time.time() - time_start)


def W_TA_time():
    A_da = self.dataset.ds.associations
    d_a = A_da.toarray()
    # print(len(A_da.T))
    le = len(d_a.T)
    author_scores = [0 for i in range(0, le)]
    start_time = time.time()
    for i, doc in enumerate(document_sorting_indices):
        authors_index = np.flatnonzero(d_a[doc])
        for idx in authors_index:
            author_scores[idx] += 1 / (i + 1)
    candidate_score = np.sort(author_scores)
