# documents_representations.py

import impact_query_expert_finding.main.language_models
import impact_query_expert_finding.data.config
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
# datasets_versions = ["VT"]
datasets_versions = ["V1"]
# datasets_types = ["dataset_associations", "dataset_cleaned"]
datasets_types = ["dataset_cleaned"]

# for dv in datasets_versions:
#     for dt in datasets_types:
#         input_dir = os.path.join(current_folder, 'output/', 'data/', dv, dt)
#         output_dir = os.path.join(input_dir, 'documents_representations')
#         parameters = {
#             'output_dir': output_dir,
#             'input_dir': input_dir,
#             'dataset': "aminer"
#         }
#         impact_query_expert_finding.main.language_models.run(parameters)

# for dv in datasets_versions:
#     for dt in datasets_types:
#         # input_dir = os.path.join(current_folder, 'output/', 'data/', dv, dt)
#         input_dir = "/ddisk/lj/LExR/data/VT/" + dt
#         output_dir = os.path.join(input_dir, 'documents_representations')
#         parameters = {
#             'output_dir': output_dir,
#             'input_dir': input_dir,
#             'dataset': 'LExR'
#         }
#         impact_query_expert_finding.main.language_models.run(parameters)

for dv in datasets_versions:
    for dt in datasets_types:
        # input_dir = os.path.join(current_folder, 'output/', 'data/', dv, dt)
        input_dir = "/ddisk/lj/ACM/data/V1/" + dt
        output_dir = os.path.join(input_dir, 'documents_representations')
        parameters = {
            'output_dir': output_dir,
            'input_dir': input_dir,
            'dataset': "acm"
        }
        impact_query_expert_finding.main.language_models.run(parameters)