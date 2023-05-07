import impact_query_expert_finding.main.evaluate
import impact_query_expert_finding.main.topics
import os

current_folder = os.path.dirname(os.path.abspath(__file__))

algorithms = ["vote"]
documents_representations = ["tfidf"]
datasets_version = ["V1"]
dataset_type = "dataset_association"
query_types = ["documents"]
# datasets_version = ["V1"]
# algorithms = ["vote"]
# documents_representations = ["tfidf"]
# dataset_type = "dataset_associations"
# query_types = ["documents"]

for qt in query_types:
    for dv in datasets_version:
        input_dir = os.path.join(current_folder, 'output/', 'data/', dv, dataset_type)
        print(input_dir)
        for a in algorithms:
            for dr in documents_representations:
                output_dir = os.path.join(current_folder, 'output/',
                                          "xp_" + qt + "_" + dv + "_" + dataset_type + "_" + a + "_" + dr)
                parameters = {
                    'output_dir': output_dir,
                    'input_dir': input_dir,
                    'algorithm': a,
                    'language_model': dr,
                    'vote_technique': 'rr',
                    'eta': 0.1,
                    'seed': 0,
                    'max_queries': 50,
                    'dump_dir': output_dir
                }
                if qt is "documents":
                    impact_query_expert_finding.main.evaluate.run(parameters)
                if qt is "topics":
                    impact_query_expert_finding.main.topics.run(parameters)
