import scipy.sparse
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import scipy

logger = logging.getLogger()


class Model:

    def __init__(self):
        pass

    def fit(self, A_da, A_dd, T):
        self.A_da = A_da

        # bert_model = SentenceTransformer('bert-base-nli-stsb-wkpooling')
        # print(bert_model.get_max_seq_length());
        # print(bert_model.get_max_seq_length());
        # "/tmp/pycharm_project_463/scripts/continue/output/training_bert-base-uncased-2020-09-15_16-38-01"
        # bert_model = SentenceTransformer("/home/lj/tmp/pycharm_project_463/scripts/continue/output/training_bert_base_uncased-2020-09-15_16-38-01")
        # bert_model = SentenceTransformer("/home/lj/tmp/pycharm_project_463/scripts/continue/output/training_nli_sts_dblproberta-base-nli-stsb-mean-tokens")
        #bert_model = SentenceTransformer("/home/lj/tmp/pycharm_project_463/scripts/continue/output/training_dblp1bert-base-uncased")
        # bert_model = SentenceTransformer("/home/lj/tmp/pycharm_project_463/tests/output/training_nli_allenai-scibert_scivocab_uncased-2020-09-21_15-36-32")
        bert_model = SentenceTransformer("/home/lj/tmp/pycharm_project_463/tests/output/training_stsbenchmark_continue_training-sci_bert_nil-2020-09-22_08-50-29")
        # bert_model = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")
        bert_model._first_module().max_seq_length = 500

        self.docs_vectors = bert_model.encode(T)

    def predict(self, d, mask=None):
        query_vector = self.docs_vectors[d]
        documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T))
        # documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T).A)
        documents_sorting_indices = documents_scores.argsort()[::-1]
        document_ranks = documents_sorting_indices.argsort() + 1

        # Sort scores and get ranks // Rank sore = 1 / docuement_ranks
        candidates_scores = np.ravel(
            self.A_da.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
                axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b

        if mask is not None:
            candidates_scores = candidates_scores[mask]
        return candidates_scores

# import impact_query_expert_finding.models.random_model
# import impact_query_expert_finding.models.panoptic_model
# import impact_query_expert_finding.models.voting_model_LExR
# import impact_query_expert_finding.models.voting_model
# import impact_query_expert_finding.models.bipartite_propagation_model
#
# __all__ = [
#     'impact_query_expert_finding.models.random_model',
#     'impact_query_expert_finding.models.panoptic_model',
#     'impact_query_expert_finding.models.voting_model',
#     'impact_query_expert_finding.models.voting_model_LExR',
#     'impact_query_expert_finding.models.bipartite_propagation_model'
# ]
