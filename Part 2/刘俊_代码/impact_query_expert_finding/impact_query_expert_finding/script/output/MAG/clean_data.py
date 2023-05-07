import json
import itertools
import random

import json
import impact_query_expert_finding.data.config
import os
import impact_query_expert_finding.data.sets
import impact_query_expert_finding.evaluation.visual

from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
import random

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def topic_authors(fq, ft, fo, fe):
    f_i_qre = open(fq, 'r')
    f_i_top = open(ft, 'r')
    f_o_aut = open(fo, 'w')
    f_o_exp = open(fe, 'w')
    i2t = {}
    authors = {}

    for topic in f_i_top:
        str = topic.split('\t')
        i2t[str[0]] = str[1].split('\n')[0]
    count = 0

    experts = list()
    for con in f_i_qre:
        split = con.split("\t")
        if i2t[split[0]] not in authors:
            authors[i2t[split[0]]] = list()
        if split[3] == '3\n' or split[3] == '2\n':
            count += 1
            authors[i2t[split[0]]].append(split[2])
            experts.append(split[2])

    json.dump(list(set(experts)), f_o_exp)
    json.dump(authors, f_o_aut)
    return fe


# topic_authors()


# 作者的文章.
def author_documents():
    dir = "/ddisk/lj/LExR"
    fi = open(dir + '/data/V2/LExR_3_documents.json', 'r')
    fo = open(dir + '/continue/experts_3.json', 'r')
    fe = open(dir + '/LExR_23_documents.json', 'w')
    fad = open(dir + '/data/V2/author_document.json', 'w')
    author = []
    for con in fo:
        author = json.loads(con)

    author_document = {}
    i = 0
    for con in fi:
        data = json.loads(con)
        for a in author:
            if a in data['authors']:
                # print(a)
                if a not in author_document:
                    author_document[a] = list()
                author_document[a].append(i)
                # json.dump(data, fe)
                # fe.write('\n')
        i += 1
    json.dump(author_document, fad)
    fi.close()


##
def document_authors():
    dir = "/ddisk/lj/LExR"
    fi = open(dir + '/data/V2/LExR_3_documents.json', 'r')
    fad = open(dir + '/data/V2/document_author.json', 'w')
    document_author = {}
    i = 0
    for con in fi:
        data = json.loads(con)
        if i not in document_author:
            document_author[i] = list()
        document_author[i].append(data['authors'])
        i += 1
    json.dump(document_author, fad)
    fi.close()
    fad.close()


def build_triplet():
    dir = "/ddisk/lj/LExR"
    fad = open(dir + '/data/V2/author_document.json', 'r')
    fda = open(dir + '/data/V2/document_author.json', 'r')
    # trip = open(dir + "data/V2/triplet.json", 'w')

    author_documents = {}
    document_authors = {}

    for con in fad:
        author_documents = json.loads(con)
    for con in fda:
        document_authors = json.loads(con)

    num = 0
    les2hud = 0
    mor2hud = 0
    count = 0
    for k, v in author_documents.items():
        num = max(num, len(v))
        count += len(v)
        if len(v) < 200:
            les2hud += 1
        else:
            mor2hud += 1
        # print(len(v))
    print(count)
    num = 0
    for k, v in document_authors.items():
        num = max(num, len(v[0]))
        if len(v) < 200:
            les2hud += 1
        else:
            mor2hud += 1
        # print(len(v[0]))
    # print(num)

    # print(les2hud)
    # print(mor2hud)


# build_triplet()


def load_author_document():
    dir = "/ddisk/lj/LExR"
    fad = open(dir + '/data/V2/author_document.json', 'r')
    for con in fad:
        author = json.loads(con)
        print(author["5768000922241088"])
        # print(len(author))


def document_contain_author(document_in, documents_out, authors_in):
    f_i_doc = open(document_in, 'r')
    f_i_auth = open(authors_in, 'r')
    f_o_doc = open(documents_out, 'w')

    document_contain = {}
    authors = []
    for auth in f_i_auth:
        authors = json.loads(auth)

    for doc in f_i_doc:
        data = json.loads(doc)
        for author in authors:
            if author in data['authors']:
                json.dump(data, f_o_doc)
                f_o_doc.write('\n')

    f_i_doc.close()
    f_i_auth.close()
    f_o_doc.close()
    return documents_out


def author_to_document(author_in, document_in, file_out):
    f_i_auth = open(author_in, 'r')
    f_i_doc = open(document_in, 'r')
    f_o_doc = open(file_out, 'w')
    authors = []
    for con in f_i_auth:
        authors = json.loads(con)
    # print(authors)
    author_document = {}
    i = 0
    for con in f_i_doc:
        data = json.loads(con)
        for a in authors:
            if a in data['authors']:
                if a not in author_document:
                    author_document[a] = list()
                author_document[a].append(i)
        i += 1
    json.dump(author_document, f_o_doc)
    f_i_doc.close()
    f_i_auth.close()
    f_o_doc.close()


def document_to_author(document_in, file_out):
    f_i_doc = open(document_in, 'r')
    f_o_doc = open(file_out, 'w')
    document_author = {}
    i = 0
    for con in f_i_doc:
        data = json.loads(con)
        if i not in document_author:
            document_author[i] = list()
        document_author[i].append(data['authors'])
        i += 1
    json.dump(document_author, f_o_doc)
    f_i_doc.close()
    f_o_doc.close()


def statistics(a2d, d2a):
    f_i_a2d = open(a2d, 'r')
    f_i_d2a = open(d2a, 'r')

    author_documents = {}
    document_authors = {}

    for con in f_i_a2d:
        author_documents = json.loads(con)
    for con in f_i_d2a:
        document_authors = json.loads(con)

    max_doc_num = 0
    author_num = 0
    total_num = 0
    name = ""
    for k, v in author_documents.items():
        author_num += 1
        total_num += len(v)
        if len(v) > max_doc_num:
            max_doc_num = len(v)
            name = k

    max_author_num = 0
    paper_num = 0
    index = ""
    for k, v in document_authors.items():
        paper_num += 1
        if len(v[0]) > max_author_num:
            # print(v)
            max_author_num = len(v[0])
            index = k

    print("Expert_num: " + str(author_num))
    print("Average: " + str(total_num / author_num))
    print("Doument_num: " + str(paper_num))
    print(name + ": has_max_document_num: " + str(max_doc_num))
    print(index + ": has_max_author_num: " + str(max_author_num))


def build(input_dir, output_dir, version, documents, find_qrels="LExR-find-qrels", find_topic="LExR-find-topics"):
    # f_doc = open(input_dir + documents, 'r')
    # f_e = open(output_dir + version, 'rw')
    author_dir = output_dir + version + "author.json"
    expert_dir = output_dir + version + "expert.json"

    topic_authors(input_dir + find_qrels, input_dir + find_topic, author_dir, expert_dir)

    paper_dir = output_dir + version + "paper.json"
    document_contain_author(input_dir + "smaller.json", paper_dir, expert_dir)

    author_to_document(expert_dir, paper_dir, output_dir + version + "author_to_document.json")

    document_to_author(paper_dir, output_dir + version + "document_to_author.json")


# p authors[topic].append([2])

def build_triplet(author_in, version, document_in, limit):
    f_i_a2d = open(author_in + version + document_in, 'r')
    f_o_doc = open(output_dir + version + "triplets_" + str(limit) + ".json", 'w')

    author_documents = {}
    for con in f_i_a2d:
        author_documents = json.loads(con)

    vis = [0] * 41201
    triple_list = list()

    print("author_d" + str(len(author_documents.items())))
    for a, ds in author_documents.items():
        for i in range((len(ds) - 1) % limit):
            triplet = ""
            if vis[ds[i]] == 0:
                mod = min(limit, len(ds) - 1)
                rand = random.randint(0, 41199)
                while (rand in ds):
                    rand = random.randint(0, 41199)
                triplet = str(ds[i]) + "\t" + str(ds[i + 1 % mod]) + "\t" + str(rand)
                vis[ds[i]] = 1
                # vis[ds[i + 1 % mod]] = 1
                triple_list.append(triplet)

    json.dump(triple_list, f_o_doc)


def get_triples(input_dir):
    f_i_t = open(input_dir, 'r')
    triple_list = []
    for con in f_i_t:
        triple_list = json.loads(con)
        # split = triple_list[0].split('\t')
        # print(triple_list[0])
        # print(split[1])
        # print(triple_list[0][0])
        print(len(triple_list))


# get_triples(output_dir + version + "triplets_80.json")

# output_dir = "/ddisk/lj/LExR/data/"

def train_triples(init_model_dir, init_model_name, input_dir, version, data_type, triples_type):
    path = init_model_dir
    train_batch_size = 16
    num_epochs = 4

    model_save_path = output_dir + version + "LExR_triples" + str(triples_type)
    dataset = impact_query_expert_finding.data.sets.DataSet("LExR")
    # "/ddisk/lj/LExR/data/VT/dataset_full"
    print(input_dir + version + data_type)
    # dataset.load_LExR("/ddisk/lj/LExR/data/VT/dataset_full")
    dataset.load_LExR(input_dir + version + data_type)
    T = dataset.ds.documents

    model = SentenceTransformer(path + init_model_name)

    f_i_t = open(input_dir + version + "triplets_" + str(triples_type) + ".json", 'r')

    train_samples = list()
    for con in f_i_t:
        triple_list = json.loads(con)
        for triple in triple_list:
            row = triple.split('\t')
            train_samples.append(
                InputExample(texts=[str(T[int(row[0])]), str(T[int(row[1])]), str(T[int(row[2])])], label=0))

    train_dataset = SentencesDataset(train_samples, model=model)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.TripletLoss(model=model)

    evaluator = TripletEvaluator.from_input_examples(train_samples, name='dev')

    warmup_steps = int(len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)

    print(model_save_path)


import csv


def build_data_json(train_data, document_in, file_out, file_name):
    f_i_doc = open(document_in + file_name, 'r')
    f_o1_doc = open(file_out + "document_references.json", 'w')
    f_o2_doc = open(file_out + "document_authors.json", 'w')
    csv_reader = csv.reader(open(train_data, 'r'))

    document_documents = {}
    document_authors = {}
    data = {}
    for con in f_i_doc:
        data = json.loads(con)
        # print(data)
        # print(data['ca4e5f779a0bf6f4edafd5144c1ca887c9878ee7']['references'])

    write_egde_num = 0
    reference_edge_num = 0
    for i, row in enumerate(csv_reader):
        if i == 0:
            continue
        else:
            # split = row.split(",")
            if row[0] in data:
                document_documents[row[0]] = data[row[0]]['references']
                document_authors[row[0]] = data[row[0]]['authors']
                refernece = data[row[0]]['references']
                write_author = data[row[0]]['authors']
                if refernece is not None:
                    reference_edge_num += len(refernece)
                write_egde_num += len(write_author)

    print("document_author_edge_num: " + str(write_egde_num))
    print("document_references_edge_num :" + str(reference_edge_num))
    # json.dump(document_documents, f_o1_doc)
    # json.dump(document_authors, f_o2_doc)
    f_i_doc.close()
    f_o1_doc.close()
    f_o2_doc.close()


input_dir = "/ddisk/lj/specter/scidocs/data/"
output_dir = "/ddisk/lj/specter/scidocs/data/mag/"

build_data_json(input_dir + "mag/" + "train.csv", input_dir, output_dir, "paper_metadata_mag_mesh.json")
