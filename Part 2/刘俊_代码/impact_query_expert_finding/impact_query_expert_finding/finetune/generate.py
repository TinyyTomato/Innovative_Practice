# encoding: utf-8
import impact_query_expert_finding.data.sets as  sets
import impact_query_expert_finding.data.io as io
import impact_query_expert_finding.data.io_aminer as aminer
import impact_query_expert_finding.data.config
import impact_query_expert_finding.data.sets
import impact_query_expert_finding.evaluation.visual
import impact_query_expert_finding.tools.graphs
import impact_query_expert_finding.finetune.train
import impact_query_expert_finding.script.document_embedding
import impact_query_expert_finding.script.experiment_acm

import os
import patoolib
import zipfile
import pkg_resources
import urllib.request
import random
import numpy as np

# from Queue import Queue


parameters = {
    'output_dir': "/ddisk/lj/DBLP/data",
    'dump_dir': "/ddisk/lj/DBLP/data_info"
}


# config_path = pkg_resources.resource_filename("impact_query_expert_finding", 'conf.yml')
# config = impact_query_expert_finding.data.config.load_from_yaml(config_path)
# data_folder = os.path.join(parameters["output_dir"], "V1")
# input_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v1"])

# dataset = sets.DataSet("aminer")
# data_path = "/ddisk/lj/DBLP/data/V1/dataset_associations"
# dataset.load(data_path)
# documents = aminer.load_papers(input_file)

# experts = dataset.ds.candidates
# A_dd = dataset.ds.citations
# A_da = dataset.ds.associations
# # A_da_weight = dataset.ds.associations_weight
# A_ad = A_da.transpose()
# T = dataset.ds.documents
#
# G_ta = dataset.gt.associations

# print(len(T))
# print(A_ad.shape)

# data_version = "V1"
#
# sample_rate = [0.2, 0.3, 0.4]
# rate = ["02", "03", "04"]
# work_dir = '/ddisk/lj/DBLP/data/'
#
# communities_dir = "/ddisk/lj/triples"
# communities_name = data_version + "_" + "03" + "_commities"
#
# triples_dir = "/ddisk/lj/triples"
# triples_name = data_version + "_" + "03" + "_triples"
#
# meta_path = "PAP"
# encoder_name = data_version + "_" + "03" + "_" + meta_path
#
# embedding_path = work_dir + data_version + "/dataset_associations/embedding"
# model_path = work_dir + data_version + "/dataset_associations/output"

class Generator:
    def __init__(self, version, work_dir, communities_path, sample_rate, meta_path, dataset):
        self.version = version
        self.work_dir = work_dir
        self.k = 15
        self.strategy = []
        self.communities_path = communities_path
        self.sample_rate = sample_rate
        self.meta_path = meta_path
        self.positive = []
        self.seed_topic = list()
        self.dataset = dataset

    def k_core_search_info(self):
        for a in A_da.getrow(3140):
            # print(a)
            for b in np.nonzero(a.toarray()[0])[0]:
                print(b)

    def get_seed_paper(self, sample_rate, T):
        self.positive = [[] for i in range(self.dataset.ds.associations.shape[-1])]

        seed_paper = list()
        # self.seed_topic = list()
        # n = sample_rate * len(T)
        # while (n > 0):
        #     rand = random.randint(0, len(T) - 1)
        #     seed_paper.append(rand)
        #     n -= 1
        for i, t in enumerate(self.dataset.gt.topics):
            experts_indices = self.dataset.gt.associations[i, self.dataset.gt.experts_mask].nonzero()[1]
            # experts_booleans = np.zeros(len(self.dataset.gt.experts_mask))
            # experts_booleans[experts_indices] = 1
            # self.labels_y_true.append(experts_booleans)
            documents_indices = np.unique(
                self.dataset.ds.associations[:, self.dataset.gt.experts_mask[experts_indices]].nonzero()[0])
            maxq = len(documents_indices)
            # under topic i , all the documents.
            self.positive[i] = documents_indices
            # if self.max_queries is not None:
            #     np.random.shuffle(documents_indices)
            #     maxq = min(self.max_queries, maxq)
            for j in range(int(maxq * sample_rate)):
                d = documents_indices[j]
                seed_paper.append(d)
                self.seed_topic.append(i)
                # self.queries.append(d)
                # self.labels.append(i)
                # self.queries_experts.append(self.dataset.ds.associations[d, :].nonzero()[1])
        return seed_paper

    def get_k_core_commity_PAP(self, seed_paper, k, A_da, A_ad):
        Tque = set()
        Qque = set()
        N = set()
        S = set()
        vis = A_da.shape[0] * [0]
        # vis = [0 for i in range(A_da.shape[0])]
        # print(len(vis))
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

    def get_k_core_commity_PTP(self, seed_paper, k, G_ta, A_ad):
        Tque = set()
        Qque = set()
        N = set()
        S = set()

        A_td = G_ta.dot(A_ad)
        A_dt = A_td.transpose()
        # print(A_td.shape)
        vis = A_dt.shape[0] * [0]
        Phi = [set() for i in range(A_dt.shape[0])]

        Tque.add(seed_paper)
        S.add(seed_paper)
        vis[seed_paper] = 1
        while len(Tque) > 0:
            doc = Tque.pop()
            for topic in np.nonzero(A_dt[doc].toarray()[0])[0]:
                for t_doc in np.nonzero(A_td[topic].toarray()[0])[0]:
                    if vis[t_doc] == 0:
                        vis[t_doc] = 1
                        Phi[doc].add(t_doc)
                        S.add(t_doc)

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

        for topic in np.nonzero(A_dt[seed_paper].toarray()[0])[0]:
            for t_doc in np.nonzero(A_td[topic].toarray()[0])[0]:
                N.add(t_doc)

        commity = S.union(N)
        # print(len(commity))
        return commity

    def get_k_core_commity_PCP(self, seed_paper, k, A_dd):
        Tque = set()
        Qque = set()
        N = set()
        S = set()

        vis = A_dd.shape[0] * [0]
        Phi = [set() for i in range(A_dd.shape[0])]

        Tque.add(seed_paper)
        S.add(seed_paper)
        vis[seed_paper] = 1
        while len(Tque) > 0:
            doc = Tque.pop()
            for c_doc in np.nonzero(A_dd[doc].toarray()[0])[0]:
                if vis[c_doc] == 0:
                    vis[c_doc] = 1
                    Phi[doc].add(c_doc)
                    S.add(c_doc)

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

        for c_doc in np.nonzero(A_dd[seed_paper].toarray()[0])[0]:
            N.add(c_doc)

        commity = S.union(N)
        # print(len(commity))
        return commity

    def get_commities(self, seed_papers, k, A_da, A_ad, communities_dir, communities_name, meta_path):
        commities = list()
        n = str(len(seed_papers))
        for i, seed in enumerate(seed_papers):
            if meta_path is "PAP":
                commity = self.get_k_core_commity_PAP(seed, k, A_da, A_ad)
                commities.append(list(commity))
            if meta_path is "PTP":
                commity = self.get_k_core_commity_PTP(seed, k, A_da, A_ad)
                commities.append(list(commity))
            if meta_path is "PCP":
                commity = self.get_k_core_commity_PCP(seed, k, A_ad)
                commities.append(list(commity))

        print("commities num:", len(commities))
        # io.save_as_json(communities_dir, communities_name, commities)
        return commities

    def get_commities_muti(self, seed_papers, k, A_da, A_ad, communities_dir, communities_name, meta_path):
        dataset = self.dataset
        A_da = dataset.ds.associations
        A_ad = A_da.transpose()
        A_dd = dataset.ds.citations
        G_ta = dataset.gt.associations
        commities = list()
        n = str(len(seed_papers))
        if meta_path is "PAP":
            commities = self.get_commities(seed_papers, k, A_da, A_ad, communities_dir, communities_name, meta_path)
        if meta_path is "PTP":
            commities = self.get_commities(seed_papers, k, G_ta, A_ad, communities_dir, communities_name, meta_path)
        if meta_path is "PCP":
            commities = self.get_commities(seed_papers, k, A_dd, A_dd, communities_dir, communities_name, meta_path)
        if meta_path is "PAPPTP":
            for i, seed in enumerate(seed_papers):
                commity_pap = self.get_k_core_commity_PAP(seed, k, A_da, A_ad)
                commity_ptp = self.get_k_core_commity_PTP(seed, k, G_ta, A_ad)
                commity = commity_pap & commity_ptp
                commities.append(list(commity))
        if meta_path is "PCPPTP":
            for i, seed in enumerate(seed_papers):
                commity_pcp = self.get_k_core_commity_PCP(seed, k, A_dd)
                commity_ptp = self.get_k_core_commity_PTP(seed, k, G_ta, A_ad)
                commity = commity_pcp & commity_ptp
                commities.append(list(commity))
        if meta_path is "PAPPCP":
            for i, seed in enumerate(seed_papers):
                commity_pap = self.get_k_core_commity_PAP(seed, k, A_da, A_ad)
                commity_pcp = self.get_k_core_commity_PCP(seed, k, A_dd)
                commity = commity_pap & commity_pcp
                commities.append(list(commity))
        if meta_path is "PAPPCPPTP":
            for i, seed in enumerate(seed_papers):
                commity_pap = self.get_k_core_commity_PAP(seed, k, A_da, A_ad)
                commity_pcp = self.get_k_core_commity_PCP(seed, k, A_dd)
                commity_ptp = self.get_k_core_commity_PTP(seed, k, G_ta, A_ad)
                commity = (commity_pap | commity_pcp) & commity_ptp
                commities.append(list(commity))
        return list(commities)

    # commities = get_k_core_commity([], documents, 4)
    def sample_triples(self, seed_papers, comunities, T, triples_dir, triples_name, core_k, negative_stratage=None):
        triplets = list()
        all = range(0, len(T) - 1)

        for i, seed in enumerate(seed_papers):
            negative = list(set(all) - set(comunities[i]) - set(self.positive[self.seed_topic[i]]))
            for pos in comunities[i][0:min(core_k, len(comunities[i]))]:
                # neg3 = np.random.choice(negative)
                # neg1 = np.random.choice(negative)
                neg2 = np.random.choice(negative)
                # triplets.append([seed, pos, neg3])
                # triplets.append([seed, pos, neg1])
                triplets.append([pos, seed, neg2])
        print("triples sample num: ", len(triplets))
        io.save_as_json(triples_dir, triples_name, triplets)
        return triplets

    def sample_triples_near_negatives(self, seed_papers, comunities, T, triples_dir, triples_name,
                                      core_k, negative_stratage=None):
        triplets = list()
        all = range(0, len(T) - 1)
        for i, seed in enumerate(seed_papers):
            negative = list(set(all) - set(comunities[i]) - set(self.positive[self.seed_topic[i]]))
            for pos in comunities[i][0:core_k]:
                neg = np.random.choice(negative)
                triplets.append([seed, pos, neg])
        print("triples sample num: ", len(triplets))
        io.save_as_json(triples_dir, triples_name, triplets)
        return triplets

# sps = get_seed_paper(sample_rate=sample_rate, T=T)
# # if io.load_as_json("/ddisk/lj/triples/" + "V1_commities")

# cms = get_commities(sps, 20)
#
# triplets = sample_triples(sps, cms, T)
#
# io.save_as_json(triples_dir, triples_name, triplets)

# impact_query_expert_finding.finetuning.train.train(config=None, triples_dir=triples_dir, triples_name=triples_name,
# version = data_version, encoder_name = encoder_name)


# impact_query_expert_finding.script.document_embedding.save_embedding(save_dir=embedding_path, model_dir=model_path,
# model_name=encoder_name)

# impact_query_expert_finding.script.experiment_acm.run(version=data_version, model_dir=embedding_path,
#                                                       model_name=encoder_name, work_dir=work_dir)

# print(G_ta.shape)
# print(A_ad.shape)
# get_k_core_commity_PTP(1, 1000, G_ta, A_ad)
# get_k_core_commity_PCP(1000, 1, A_dd)
# s1 = {1, 2, 3, 4}
# s2 = {3, 4, 5, 6}
# sn = s1 & s2
# print(sn)
