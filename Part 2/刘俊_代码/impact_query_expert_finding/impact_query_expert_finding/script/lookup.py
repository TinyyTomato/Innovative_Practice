import numpy as np
import os
import gensim
import numpy as np
import scipy.sparse
import time
import re

stop_words = {'an', 'through', 'own', 'too', 'again', 'nor', 'doesn', "didn't", 'whom', 'couldn', 'shouldn', 'him',
              'after', "weren't", 'or', 'itself', "shouldn't", "isn't", 'to', 'needn', "hasn't", 'of', 'her', 'why',
              'against', 'should', 'themselves', 'didn', "won't", 'the', 'our', 'here', 'we', 'does', 'at', 'during',
              'it', 'such', 'for', 'about', 'wasn', 'once', 'did', 'out', 'but', 'their', 'myself', 'will', 'herself',
              'below', "needn't", "should've", 'my', 'no', 'down', 'his', 'they', 'is', 'so', "shan't", "aren't",
              "it's", 'each', 'into', "you've", 'himself', 'above', 'aren', 'wouldn', 'most', 'this', "wasn't",
              'theirs', "mustn't", 'm', 'both', "wouldn't", 'mustn', 'because', 'all', "couldn't", "mightn't", "you'd",
              'with', 'other', 'am', 's', 'll', "hadn't", 'won', 're', "don't", 'were', 'me', 'these', 'off', "you'll",
              'as', 'over', 'while', 'haven', 'on', 'then', "haven't", 'are', 'she', 'having', 'mightn', 'yourself',
              'have', 't', 'some', 'isn', 'shan', 've', 'can', 'had', "you're", 'few', 'under', 'up', "doesn't",
              'yourselves', 'he', 'from', 'those', 'only', 'hadn', 'that', 'don', 'which', 'not', 'doing', 'yours',
              "that'll", 'been', 'in', 'where', 'hasn', 'weren', 'very', 'being', 'more', 'be', 'your', 'there', 'do',
              "she's", 'now', 'ain', 'a', 'how', 'you', 'has', 'further', 'who', 'o', 'before', 'just', 'same',
              'ourselves', 'until', 'and', 'between', 'ma', 'ours', 'what', 'them', 'y', 'hers', 'was', 'when', 'its',
              'if', 'by', 'd', 'i', 'than', 'any'}


# cat_data = np.load('.npz')
# associations = scipy.sparse.load_npz(
#     "D:\Just\impact_query_expert_finding\impact_query_expert_finding\script\output\data\V1\dataset_associations\df_associations.npz")
# print(associations.shape)
# print(associations)
#
# citations = scipy.sparse.load_npz(
#     "D:\Just\impact_query_expert_finding\impact_query_expert_finding\script\output\data\V1\dataset_associations\df_citations.npz")
# print("citations ++++++++++++++++++++++++++++++++++++++++  citations")
# print(citations)
#
# gt_associations = scipy.sparse.load_npz(
#     "D:\Just\impact_query_expert_finding\impact_query_expert_finding\script\output\data\V1\dataset_associations\gt_associations.npz")
# print("gt  ++++++++++++++++++++++++++++++++++++++++  gt")
# print(gt_associations)
def tokenize(text, input_dir, max_num_words=3, grams=None):
    tokens = preprocess_text([text])[0]
    if grams == None:
        grams = gensim.models.phrases.Phraser.load(os.path.join(input_dir, "phraser"))
    for i in range(max_num_words - 1):
        tokens = grams[tokens]
    return tokens


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def preprocess_text(stream):
    tokens_stream = [
        gensim.utils.simple_preprocess(cleanhtml(gensim.utils.decode_htmlentities(t))
                                       , deacc=True, min_len=1, max_len=100) for t in
        stream]
    for i, tokens in enumerate(tokens_stream):
        tokens_stream[i] = [j for j in tokens if j not in stop_words]
    return tokens_stream


input_dir = "D:\Just\impact_query_expert_finding\impact_query_expert_finding" \
            "\script\output\data\V1\dataset_associations\documents_representations"

# grams = gensim.models.phrases.Phraser.load(os.path.join(input_dir, "phraser"))
# print(grams)
#
# corpus_index = 0
# dictionary = gensim.corpora.Dictionary.load(
#     os.path.join(input_dir, "dictionary"))
# print(dictionary.doc2bow(tokenize("", input_dir, grams=)))
# self.corpus_index[self.dictionary.doc2bow(tokenize(input_text, self.input_dir, grams=self.grams))])

corpus_index = gensim.similarities.docsim.SparseMatrixSimilarity.load(
    os.path.join(input_dir, "corpus_index"))
dict = [(7, 1)]
print(max(corpus_index[dict]))
