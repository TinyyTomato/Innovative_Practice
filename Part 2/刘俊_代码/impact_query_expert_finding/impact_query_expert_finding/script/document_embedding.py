# encoding: utf-8
import impact_query_expert_finding.main.evaluate
import impact_query_expert_finding.main.topics
import impact_query_expert_finding.data.io
import os
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import pickle
import impact_query_expert_finding.data.io as io
import impact_query_expert_finding.data.sets as  sets
import numpy as np
import faiss
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
current_folder = os.path.dirname(os.path.abspath(__file__))
faiss.omp_set_num_threads(8)


def save_embedding(model_list=None, save_dir=None, model_dir=None, model_name=None, dataset=None):
    bert_model = SentenceTransformer(os.path.join(model_dir, model_name))
    embedding_docs_vectors = normalize(bert_model.encode(dataset.ds.documents), norm='l2', axis=1)
    io.save_as_json(save_dir, model_name, embedding_docs_vectors)
    build_PG_index(save_dir=save_dir, model_dir=model_dir, model_name=model_name, dataset=dataset)
    # build_NSG_index_overHead(save_dir=save_dir, model_dir=model_dir, model_name=model_name, dataset=dataset)


def get_embedding(model_list=None, save_dir=None, model_dir=None, model_name=None, dataset=None):
    bert_model = SentenceTransformer(os.path.join(model_dir, model_name))
    embedding_docs_vectors = normalize(bert_model.encode(dataset.ds.documents), norm='l2', axis=1)
    return embedding_docs_vectors


def build_PG_index(model_list=None, save_dir=None, model_dir=None, model_name=None, dataset=None):
    embeddings = io.load_as_json(save_dir, model_name)
    time_start = time.time()
    # faiss_index = faiss.IndexNSGFlat(len(embeddings[0]), 16)
    faiss_index = faiss.IndexFlatL2(len(embeddings[0]))
    faiss_index.add(np.array(embeddings, dtype='float32'))
    faiss.write_index(faiss_index, os.path.join(save_dir, model_name + "PG"))
    print("Dataset : ", model_name, " size : ", len(embeddings), "index build time: ", time.time() - time_start, "s")


def build_NSG_index(model_list=None, save_dir=None, model_dir=None, model_name=None, dataset=None):
    embeddings = io.load_as_json(save_dir, model_name)
    faiss_index = faiss.IndexHNSWFlat(len(embeddings[0]), 16)
    faiss_index.add(np.array(embeddings, dtype='float32'))
    faiss.write_index(faiss_index, os.path.join(save_dir, model_name + "HNSW"))

def build_NSG_index_overHead(model_list=None, save_dir=None, model_dir=None, model_name=None, dataset=None):
    embeddings = io.load_as_json(save_dir, model_name)
    for i in range(0, 6):
        embeddings = np.r_[embeddings, embeddings]
        print(len(embeddings), " ", len(embeddings[0]))
        query_vector_emb = embeddings[0]
        time_brute = time.time()
        documents_scores = np.squeeze(query_vector_emb.dot(embeddings.T))
        np.sort(documents_scores)
        print("Dataset : ", model_name, " size : ", len(embeddings), "brute query time: ", time.time() - time_brute,
              "s")

        time_start = time.time()
        # faiss_index = faiss.IndexNSGFlat(len(embeddings[0]), 8)
        # faiss_index.train()
        faiss_index = faiss.IndexHNSWFlat(len(embeddings[0]), 16)
        faiss_index.add(np.array(embeddings, dtype='float32'))

        print("Dataset : ", model_name, " size : ", len(embeddings), "index build time: ", time.time() - time_start,
              "s")
        faiss.write_index(faiss_index, os.path.join(save_dir, model_name + str(i) + "PGT"))


def query_time(model_list=None, save_dir=None, model_dir=None, model_name=None, dataset=None):
    # faiss_index = load_PG_index(save_dir, model_name + str(4) + "PGT")
    # build_NSG_index_overHead(model_list, save_dir, model_dir, model_name, dataset)
    embeddings = io.load_as_json(save_dir, model_name)
    faiss_index = faiss.read_index(os.path.join(save_dir, model_name + str(5) + "PGT"))
    documents_num = [50, 100, 200, 500, 1000]
    for dn in documents_num:
        time_search = time.time()
        res_distance, res_index = faiss_index.search(np.array([embeddings[232]], dtype='float32'), dn)
        print("Dataset : ", model_name, " size : ", len(embeddings), "index search time: ", time.time() - time_search,
              "s")


def load_PG_index(save_dir, model_name):
    print(os.path.join(save_dir, model_name + "PG"))
    return faiss.read_index(os.path.join(save_dir, model_name + "PG"))


def load_HNSW_index(save_dir, model_name):
    return faiss.read_index(os.path.join(save_dir, model_name + "HNSW"))


def query_embedding(query):
    query = [query]
    bert_model = SentenceTransformer(dir + '/sci_bert_nil_sts')
    query_embed = normalize(bert_model.encode(query), norm="l2", axis=1)
    print(query_embed[0])


def topic_desc_embedding_save():
    describes = []
    with open("/home/lj/project/impact_query_expert_finding/script/topics_desc_bak.txt", "r") as f:
        data = f.readlines()
        for line in data:
            describes.append(line)
    return describes

# descs = topic_desc_embedding_save()
# query_embedding("boosting")
# list = [
#     '/sci_bert_nil_sts',
#     # '/LExR_triples_80'
# ]
#
# dataset_type = "dataset_associations"
# version = "V1/"
# input_dir = "/ddisk/lj/DBLP/data/" + version + dataset_type
# dataset = impact_query_expert_finding.data.sets.DataSet("acm")
# dataset.load(input_dir)
# # dir = "/home/lj/tmp/pycharm_project_463/scripts/continue/output"
# # path = "/ddisk/lj/LExR/data/" + version
#
# save_embedding(list)

# dataset = sets.DataSet("aminer")
# data_path = "/ddisk/lj/DBLP/data/V1/dataset_associations"
# dataset.load(data_path)
#
# path = "/ddisk/lj/DBLP/data/V1/dataset_associations/embedding/"
# model_path = "/ddisk/lj/DBLP/data/V1/dataset_associations/output/"
# save_embedding()
import numpy as np


