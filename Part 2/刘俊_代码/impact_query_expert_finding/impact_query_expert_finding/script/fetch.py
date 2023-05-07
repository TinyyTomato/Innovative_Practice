import impact_query_expert_finding.data.io
import impact_query_expert_finding.main.fetch_data
import impact_query_expert_finding.main.fetch_data_mag
import impact_query_expert_finding.main.fetch_data_acm
from sklearn.preprocessing import normalize
import pickle
import impact_query_expert_finding.main.evaluate
import impact_query_expert_finding.main.topics
import impact_query_expert_finding.data.io
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
import csv

import logging

import impact_query_expert_finding.main.language_models
import impact_query_expert_finding.data.config
import os
import impact_query_expert_finding.data.sets


def fetch(type=None):
    # if type is False:
    parameters = {
        'output_dir': "/ddisk/lj/DBLP/data",
        'dump_dir': "/ddisk/lj/DBLP/data_info",
        'type': type,
    }
    impact_query_expert_finding.main.fetch_data.run(parameters)

# fetch()
# else:
#     parameters = {
#         'output_dir': "/ddisk/lj/DBLP/wdata",
#         'dump_dir': "/ddisk/lj/DBLP/wdata_info",
#         'type': type,
#     }
#     impact_query_expert_finding.main.fetch_data.run(parameters)


def representation():
    # datasets_versions = ["/V1", "/V2", "/V4"]
    datasets_versions = ["/V1"]
    datasets_types = ["/dataset_associations"]
    for dv in datasets_versions:
        for dt in datasets_types:
            input_dir = "/ddisk/lj/DBLP/data/" + dv + dt
            output_dir = os.path.join(input_dir, 'documents_representations')
            parameters = {
                'output_dir': output_dir,
                'input_dir': input_dir,
                'dataset': "aminer"
            }
            print(output_dir)
            impact_query_expert_finding.main.language_models.run(parameters)

# fetch()
# representation()

def get_embedding():
    path_lj = "/ddisk/lj"
    model_list = ['/doc_doc_sci_bert_triples_nil_sts']
    data_type = ["/dataset_associations"]
    data_set = ["/DBLP"]
    data_version = ["/V2"]
    data_path = path_lj + data_set[0] + "/data" + data_version[0] + data_type[0]
    embedding_path = data_path + "/embedding"
    model_path = data_path + data_set[0] + "/data" + data_version[0] + "model"

    # data_path = "/ddisk/lj/DBLP/data/V2/dataset_associations/"
    # embedding_path = "/ddisk/lj/DBLP/data/V2/dataset_associations/embedding"
    # model_path = "/ddisk/lj/DBLP/data/V2/models"

    dataset = impact_query_expert_finding.data.sets.DataSet("aminer")
    dataset.load(data_path)
    for name in model_list:
        impact_query_expert_finding.data.io.check_and_create_dir(embedding_path)
        with open(embedding_path + name, "wb") as f:
            bert_model = SentenceTransformer(model_path + name)
            embedding_docs_vectors = normalize(bert_model.encode(dataset.ds.documents), norm='l2', axis=1)
            pickle.dump(embedding_docs_vectors, f)


# representation()
# get_embedding()
# representation()
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def get_train_set():
    print(0)


def triplet_train():
    # A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = impact_query_expert_finding.io.load_dataset("dblp")
    dataset = impact_query_expert_finding.data.sets.DataSet("aminer")
    data_path = "/ddisk/lj/DBLP/data/V2/dataset_associations/"
    dataset.load(data_path)
    T = dataset.ds.documents
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    # You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    train_batch_size = 16
    num_epochs = 4
    model_save_path = 'output/academia_author_triplet'

    model_name = 'allenai/scibert_scivocab_uncased'

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logging.info("Read Triplet train dataset")
    train_samples = []
    dev_samples = []
    with open("./datasets/doc_triples.csv") as fIn:
        reader = csv.reader(fIn)
        for i, row in enumerate(reader):
            if i % 20 == 0:
                dev_samples.append(
                    InputExample(texts=[str(T[int(row[1])]), str(T[int(row[2])]), str(T[int(row[3])])], label=0))

            train_samples.append(
                InputExample(texts=[str(T[int(row[1])]), str(T[int(row[2])]), str(T[int(row[3])])], label=0))

    train_dataset = SentencesDataset(train_samples, model=model)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.TripletLoss(model=model)

    evaluator = TripletEvaluator.from_input_examples(dev_samples, name='dev')

    warmup_steps = int(len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################
    # model = SentenceTransformer(model_save_path)
    # test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    # test_evaluator(model, output_path=model_save_path)

    print(model_save_path)


def bulid_data_for_query():
    path_lj = "/ddisk/lj"
    data_type = ["/dataset_associations", "/dataset_cleaned"]
    data_set = ["/DBLP"]
    data_version = ["/V1", "/V2", "/V3"]
    for version in data_version:
        for type in data_type:
            data_path = path_lj + data_set[0] + "/data" + version + type
            dataset = impact_query_expert_finding.data.sets.DataSet("aminer")
            print("building : " + data_path)
            dataset.load(data_path)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            documents = dataset.ds.documents
            candidates = dataset.ds.candidates
            topics = dataset.gt.topics  # d-t  = 2 || t-d = 5

            asso = dataset.ds.associations  # d-a type = 0 || a-d = 3
            cita = dataset.ds.citations  # d-d cite = 1 || be cited = 4
            gt_asso = dataset.gt.associations

            associations = asso.tocoo()
            citations = cita.tocoo()
            gt_associations = gt_asso.tocoo()

            print("total node num : " + str(len(documents) + len(candidates) + len(topics)))
            print("documents node num : " + str(len(documents)))
            print("author node num : " + str(len(candidates)))
            print("topic node num : " + str(len(topics)))

            output_dir = data_path + "/corpus"
            impact_query_expert_finding.data.io.check_and_create_dir(output_dir)

            create_vertex(output_dir, documents, candidates, topics)
            create_edge_and_graph(output_dir, documents, candidates, associations, citations, gt_associations)

    # data_path = path_lj + data_set[0] + "/data" + data_version[0] + data_type[1]
    # dataset = impact_query_expert_finding.data.sets.DataSet("aminer")
    # dataset.load(data_path)
    # dataset.gt.candidates
    # dataset.gt.experts_mask
    # dataset.gt.associations


def create_vertex(output_dir, documents, candidates, topics):
    print("----> create vertex")
    i = 0
    # output_dir = data_path + "/corpus"
    with open(output_dir + "/vertex.txt", 'a') as f:
        while i < len(documents):
            f.write('{} {} {}'.format(i, 0, '\n'))
            i += 1
        while i < len(documents) + len(candidates):
            f.write('{} {} {}'.format(i, 1, '\n'))
            i += 1
        while i < len(documents) + len(topics) + len(candidates):
            f.write('{} {} {}'.format(i, 3, '\n'))
            i += 1


def create_edge_and_graph(output_dir, documents, candidates, associations, citations, gt_associtation):
    graph = {}
    edge_out = open(output_dir + "/edge.txt", 'a')
    graph_out = open(output_dir + "/graph.txt", 'a')
    k = 0

    candidates_docs = {}
    doc_len = len(documents)
    aut_len = len(candidates)
    print("----> create document_author and author_documents")
    #  add doc_author type 0  ---- author_doc type 3
    for (i, v_id) in enumerate(associations.row):
        if (associations.col[i] + doc_len) not in candidates_docs:
            candidates_docs[doc_len + associations.col[i]] = list()
        candidates_docs[doc_len + associations.col[i]].append(v_id)
        if v_id not in graph:
            graph[v_id] = list()
        graph[v_id].append(associations.col[i] + len(documents))
        graph[v_id].append(k)
        edge_out.write('{} {} {}'.format(k, 0, '\n'))
        k += 1
        # author_documets
        if (v_id + doc_len) not in graph:
            graph[v_id + doc_len] = list()
        graph[v_id + doc_len].append(v_id)
        graph[v_id + doc_len].append(k)
        edge_out.write('{} {} {}'.format(k, 3, '\n'))
        k += 1

    #  add_doc_doc  type 1
    print("----> create document_cite_documents and documents_becite_documents")
    for (i, v_id) in enumerate(citations.row):
        graph[v_id].append(citations.col[i])
        graph[v_id].append(k)
        edge_out.write('{} {} {}'.format(k, 1, '\n'))
        k += 1

        graph[citations.col[i]].append(v_id)
        graph[citations.col[i]].append(k)
        edge_out.write('{} {} {}'.format(k, 4, '\n'))
        k += 1

    #  now is topic-author - > documents
    #  add_doc_topic type 2
    # topics_author = dataset.gt.associations.tocoo()
    topics_author = gt_associtation
    # i index v_id topic
    print("----> create topic_documents and documents_topic")
    for i, v_id in enumerate(topics_author.row):
        # j index d_id doc_id
        if doc_len + topics_author.col[i] in candidates_docs:
            for j, d_id in enumerate(candidates_docs[doc_len + topics_author.col[i]]):
                graph[d_id].append(doc_len + aut_len + i)
                graph[d_id].append(k)
                edge_out.write('{} {} {}'.format(k, 2, '\n'))
                k += 1
                if (doc_len + aut_len + i) not in graph:
                    graph[doc_len + aut_len + i] = list()
                graph[doc_len + aut_len + i].append(d_id)
                graph[doc_len + aut_len + i].append(k)
                edge_out.write('{} {} {}'.format(k, 5, '\n'))
                k += 1

    for k, v in zip(graph.keys(), graph.values()):
        graph_out.write('{} {} {}'.format(k, v, '\n'))
    edge_out.close()
    graph_out.close()

# bulid_data_for_query()
# representation()
