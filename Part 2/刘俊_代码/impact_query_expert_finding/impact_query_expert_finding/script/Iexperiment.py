import impact_query_expert_finding.main.evaluate
import impact_query_expert_finding.main.topics
import impact_query_expert_finding.main.Itopics
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
current_folder = os.path.dirname(os.path.abspath(__file__))


# query_types = ["documents"]
# query_types = ["topics"]
# algorithms = ["vote"]
def query():
    model_list = [
        # '/sci_bert_nil_sts',
        # '/doc_doc_sci_bert_siamese',
        # '/doc_doc_sci_bert_triples',
        # '/doc_doc_sci_bert_nil_sts_triples',
        # '/doc_doc_sci_bert_triples_nil_sts',
        '/V1_PAP_SciBERT'
    ]
    documents_representations = ["embedding"]
    # datasets_version = ["V1"]  # v2 smaller
    dataset_type = "dataset_associations"
    hybird = {}
    hybird["k"] = [0.0]
    # query_types = ["documents", "topics"]
    # algorithms = ["panoptic", "vote", "bipartite_propagation"]
    # documents_representations = ["tf", "tfidf", "lsa"]
    # dataset_type = "dataset_cleaned"
    # dir = "/ddisk/lj/"
    dir = "/home/lj/project/impact_query_expert_finding/script/output/data/"
    datasets_version = ["null", "V1", "V2"]
    # query_types = ['document', 'topic_describe', 'topic']
    query_types = ['document', 'topic']
    rank_types = ['vote']

    for query_type in query_types:
        for model_name in model_list:
            for rank_type in rank_types:
                # for dr in documents_representations:
                input_dir = os.path.join(current_folder, 'output/', 'data/', datasets_version[1], dataset_type)
                print(input_dir)
                output_dir = os.path.join(current_folder, 'output/',
                                          "xp_" + "topics" + "_" + datasets_version[2]
                                          + "_" + dataset_type + model_name + "-" + str(rank_type))
                parameters = {
                    'output_dir': output_dir,
                    'input_dir': input_dir,
                    'algorithm': "vote",
                    'model_name': "/",
                    # 'model_dir': dir + datasets_version[2],
                    'model_dir': input_dir,
                    'query_type': "document",
                    'rank_type': rank_type,
                    'language_model': "tfidf",
                    'vote_technique': 'rr',
                    'eta': 0.1,
                    'seed': 0,
                    'max_queries': 50,
                    'dump_dir': output_dir,
                }
                impact_query_expert_finding.main.evaluate.run(parameters)

# def doccument_query():
#     datasets_version = "/V1"
#     dataset_type = "/dataset_associations"
#     model_name = "idne"
#     input_dir = "/ddisk/lj/DBLP/data" + datasets_version + dataset_type
#     output_dir = "ddisk/lj/DBLP/data" + datasets_version + dataset_type + "/output/" + model_name + "_doc"
#     print("input_dir > " + input_dir)
#     print("output_dir > " + output_dir)
#     parameters = {
#         'output_dir': output_dir,
#         'input_dir': input_dir,
#         'algorithm': "vote",
#         'model_name': "/embedding/V1_PAP_Emb",
#         # 'model_name': "tfidf",
#         'model_dir': input_dir,
#         'query_type': "document",
#         'rank_type': "vote",
#         'language_model': "tfidf",
#         'vote_technique': 'rr',
#         'eta': 0.1,
#         'seed': 0,
#         'max_queries': 50,
#         'dump_dir': output_dir,
#     }
#     impact_query_expert_finding.main.evaluate.run(parameters)
#
#
# doccument_query()
