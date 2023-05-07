import impact_query_expert_finding.data.sets as  sets
from impact_query_expert_finding.finetune.generate import Generator
import impact_query_expert_finding.finetune.train as trainer
import impact_query_expert_finding.script.document_embedding as saver
import impact_query_expert_finding.script.experiment_acm as evaluator
import impact_query_expert_finding.data.io as io
import impact_query_expert_finding.script.fetch as fetcher
import numpy as np
import faiss
import impact_query_expert_finding.script.document_embedding  as embedded
import time
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# from sklearn.cluster import MiniBatchKMeans

dataset = sets.DataSet("aminer")
data_path = os.path.join("/ddisk/lj/DBLP/data/", "V4", "dataset_associations")
dataset.load(data_path)
work_dir = "/ddisk/lj/DBLP/data/"
version = "V4"
data_type = "dataset_associations"


def convert_data(dataset):
    A_da = dataset.ds.associations.toarray()
    A_ad = A_da.T
    G = {'directed': False, 'graph': {'name': 'disjoint_union(, )'}, 'nodes': [], 'links': [], 'multigraph': False}
    # <class 'dict'>: {'test_removed': False, 'train_removed': False, 'target': 800, 'source': 0}
    save_dir = os.path.join(work_dir, version, data_type, "embedding")
    model_name = version + "_" + str(0.8) + "_" + "PAP" + "_" + str(15) + "core"
    embeddings = io.load_as_json(save_dir, model_name)
    embeddings = np.array(embeddings, dtype='float32')
    print(len(embeddings))

    features = []
    id_map = {}
    A_A = [[0 for i in range(len(A_ad))] for i in range(len(A_ad))]

    for i, author in enumerate(A_ad):
        id_map[str(i)] = i
        documents_index = np.flatnonzero(A_ad[i])
        if len(documents_index) == 0:
            avg_embedding = [0.001 for i in range(0, 768)]
        else:
            avg_embedding = np.mean(embeddings[documents_index], axis=0)  #
        features.append(avg_embedding)  # region features

        G['nodes'].append({'test': False, 'id': i, 'feature': avg_embedding, 'val': False})
        k = 0

        for d_idx in documents_index:
            authors_idx = np.flatnonzero(A_da[d_idx])
            for a in authors_idx:
                A_A[i][a] = 1
                if a == i: continue
                G['links'].append({'test_removed': False, 'train_removed': False, 'target': a, 'source': i})
                k += 1

    features = np.array(features)
    new_features = []
    for i, a in enumerate(A_A):
        a_i = np.flatnonzero(A_A[i])
        if (len(a_i) > 0):
            neigbor_embedding = np.mean(features[a_i], axis=0)
            new_features.append(features[i] + neigbor_embedding)
        else:
            new_features.append(features[i])

    print("save  author  features", len(new_features))
    # io.save_as_json("/ddisk/lj/DBLP/data/", "AE", new_features)
    io.save_as_json("/ddisk/lj/DBLP/data/", "AE", features)
    io.save_as_json("/ddisk/lj/DBLP/data/", "A_A", A_A)
    # io.save_as_json("/ddisk/lj/DBLP/data/", "F", features)
    # io.save_as_json("/ddisk/lj/DBLP/data/", "M", id_map)


# convert_data(dataset)
def builder_init_author_vec():
    A_da = dataset.ds.associations.toarray()
    A_ad = A_da.T
    save_dir = os.path.join(work_dir, version, data_type, "embedding")
    model_name = version + "_" + str(0.8) + "_" + "PAP" + "_" + str(15) + "core"
    embeddings = io.load_as_json(save_dir, model_name)
    embeddings = np.array(embeddings, dtype='float32')
    features = []
    author_document_cluster = []
    count = 0
    for i, author in enumerate(A_ad):
        documents_index = np.flatnonzero(A_ad[i])
        # print("init ", i, "th author vector")
        if len(documents_index) == 0:
            avg_embedding = [0.001 for i in range(0, 768)]
            features.append(avg_embedding)
            features.append(avg_embedding)
            author_document_cluster.append(documents_index)
            author_document_cluster.append(documents_index)
        elif len(documents_index) == 1:
            avg_embedding = np.mean(embeddings[documents_index], axis=0)  #
            features.append(avg_embedding)  # region features
            features.append(avg_embedding)  # region features
            author_document_cluster.append(documents_index)
            author_document_cluster.append(documents_index)
        else:
            estimator = KMeans(n_clusters=2, n_jobs=2, n_init=5)
            # fit on the whole data
            estimator.fit(embeddings[documents_index])
            zeros = np.where(estimator.labels_ == 0)[0]
            ones = np.flatnonzero(estimator.labels_)
            zeros_embedding = np.mean(embeddings[documents_index[zeros]], axis=0)
            if len(ones) == 0:
                ones_embedding = zeros_embedding
                ones = zeros
            else:
                ones_embedding = np.mean(embeddings[documents_index[ones]], axis=0)
            features.append(zeros_embedding)  # region features
            features.append(ones_embedding)  # region features
            author_document_cluster.append(documents_index[zeros])
            author_document_cluster.append(documents_index[ones])

    io.save_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding/", "AEC", np.array(features))
    io.save_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding/", "AECD", author_document_cluster)


def builder_init_author_cluster_vec():
    A_da = dataset.ds.associations.toarray()
    A_ad = A_da.T
    save_dir = os.path.join(work_dir, version, data_type, "embedding")
    model_name = version + "_" + str(0.8) + "_" + "PAP" + "_" + str(15) + "core"
    embeddings = io.load_as_json(save_dir, model_name)
    embeddings = np.array(embeddings, dtype='float32')
    features = []
    author_document_cluster = []
    author_document_cluster_embeddings = []

    for i, author in enumerate(A_ad):
        documents_index = np.flatnonzero(A_ad[i])
        if len(documents_index) == 0:
            avg_embedding = [[0.001 for i in range(0, 768)]]
            author_document_cluster_embeddings.append(avg_embedding)
            author_document_cluster.append([0])
        elif len(documents_index) == 1:
            avg_embedding = np.squeeze(np.mean(embeddings[documents_index], axis=0))  #
            author_document_cluster_embeddings.append([avg_embedding])
            author_document_cluster.append([documents_index])
        else:
            estimator = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
            # fit on the whole data
            estimator.fit(embeddings[documents_index])
            labels = estimator.labels_
            max_val = max(labels)
            i_clusters = [[] for i in range(0, max_val + 1)]
            i_clusters_embedding = [[] for i in range(0, max_val + 1)]
            for idx, type in enumerate(labels):
                i_clusters[type].append(documents_index[idx])

            for idx, docs in enumerate(i_clusters):
                avg_embedding = np.squeeze(np.mean(embeddings[docs], axis=0))
                i_clusters_embedding[idx] = avg_embedding

            author_document_cluster.append(i_clusters)
            author_document_cluster_embeddings.append(i_clusters_embedding)


    io.save_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding/", "AECM",
                    author_document_cluster_embeddings)
    io.save_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding/", "AECDM", author_document_cluster)


# builder_init_author_vec()
builder_init_author_cluster_vec()


def get_k_core_commity_APA(self, seed_paper, k, A_da, A_ad):
    Tque = set()
    Qque = set()
    N = set()
    S = set()
    vis = A_da.shape[0] * [0]

    Phi = [set() for i in range(A_da.shape[0])]

    Tque.add(seed_paper)
    S.add(seed_paper)
    vis[seed_paper] = 1
    while len(Tque) > 0:
        doc = Tque.pop()
        for author in np.nonzero(A_da[doc].toarray()[0])[0]:
            for a_doc in np.nonzero(A_ad[author].toarray()[0])[0]:
                if vis[a_doc] == 0:
                    vis[a_doc] = 1
                    Phi[doc].add(a_doc)
                    S.add(a_doc)

        if len(Phi[doc]) >= k:
            Tque.union(Phi[doc])
        else:
            Qque.add(doc)

    while len(Qque) != 0:
        v = Qque.pop()
        S.remove(v)
        for u in Phi[v].copy():
            if (u not in S): continue
            if (v in Phi[u]): Phi[u].remove(v)
            if len(Phi[u]) < k:
                Qque.add(u)

    for author in np.nonzero(A_da[seed_paper].toarray()[0])[0]:
        for a_doc in np.nonzero(A_ad[author].toarray()[0])[0]:
            N.add(a_doc)

    commity = S.union(N)
    return commity
