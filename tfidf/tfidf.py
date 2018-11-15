from gensim import corpora
from collections import defaultdict
import pickle
import math
# append import path
import os, sys
sys.path.append(os.pardir)
from calc import Calc
from addTopic import lda_value

def load_pickle(load_file_name):
    with open(load_file_name, 'rb') as f:
        advice = pickle.load(f)
    return advice

def dump_pickle(data, dump_file_name):
    with open(dump_file_name, 'wb') as f:
        pickle.dump(data, f)

def load_tfidf_model(corpus):
    from gensim import models
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus] # corpus_tfidf[0]で0番目の文書の単語のcorpusのtfidfを表示
    return corpus_tfidf

def counter(dictionary):
    texts = []
    frequency = defaultdict(int)
    for reportNo in dictionary:
        for token in dictionary[reportNo]['advice_divide_mecab']:
            frequency[token] += 1
        texts.append(advice[reportNo]['advice_divide_mecab'])
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return frequency, texts

def bag_of_word(dictionary, texts):
    return [dictionary.doc2bow(text) for text in texts]

def greater_than(array_length, n=0):
    return array_length > n

def tfidf_value_average(tfidf_sum, division_length):
    if greater_than(division_length, n=0):
        tfidf_average = tfidf_sum / division_length
    else:
        tfidf_average = 0
    return tfidf_average

def dictionary_sort_value(dictionary, desc=False):
    return sorted(dictionary.items(), key=lambda x: x[1], reverse=desc)

def add_tfidf_top_10_average(current_advice):
    tfidfs = dictionary_sort_value(current_advice['tfidf'], desc=True)
    top_10_tfidf = tfidfs[:10] # もし配列が10個未満の場合は上位10ではないが、その数で計算する
    tfidf_sum = 0
    for tfidf_tuple in top_10_tfidf:
        tfidf_sum += tfidf_tuple[1]

    tfidf_average = 0.0001
    if greater_than(len(top_10_tfidf), n=5):
        tfidf_average = tfidf_sum / len(top_10_tfidf)
    current_advice['tfidf_top_average'] = tfidf_average
    return current_advice

def gensim_tfidf(advice):
    frequency, texts = counter(advice)

    # id:単語　の形
    dictionary = corpora.Dictionary(texts)
    # corpus[0]で0番目の文書のbag-of-wordを取得できる
    corpus = bag_of_word(dictionary, texts)
    corpus_tfidf = load_tfidf_model(corpus)
    dump_pickle(corpus_tfidf, 'corpus_tfidf.pickle')

    for index, corpus_of_each_text in enumerate(corpus):
        if index % 10 == 0:
            print("進捗度: ", str(round((index+1) / len(corpus) * 100, 2)), '%')

        current_advice = advice[index]
        current_advice['tfidf'] = {}
        current_corpus_tfidf = corpus_tfidf[index]
        tfidf_vector_sum = 0
        for one_corpus in corpus_of_each_text:
            tfidf_tuple = [ct for ct in current_corpus_tfidf if ct[0] == one_corpus[0]][0]
            tfidf_word = dictionary[tfidf_tuple[0]]
            tfidf_value = tfidf_tuple[1]
            current_advice['tfidf'][tfidf_word] = tfidf_value
            tfidf_vector_sum += tfidf_value

        corpus_length = len(corpus_of_each_text)
        current_advice['tfidf_average'] = tfidf_value_average(tfidf_vector_sum, corpus_length)
        current_advice['tfidf_sum'] = tfidf_vector_sum

        add_tfidf_top_10_average(current_advice)

    return advice

def norm(dictionary, x_max, x_min, diff_max_min):
    for key in dictionary:
        current_values = dictionary[key]
        for index, word in enumerate(current_values):
            current_value = current_values[word]
            dictionary[key][word] = \
                ((current_value - x_min) / diff_max_min) + 0.001
    return dictionary

def dict_norm(dictionary, flatten_values):
    x_max = max(flatten_values)
    x_min = min(flatten_values)
    diff_max_min = x_max - x_min
    dictionary = norm(dictionary, x_max, x_min, diff_max_min)
    return dictionary

def flatten(dictionary):
    return [dictionary[key] for key in dictionary]


def gensim_bm25(advice):
    PARAM_K1 = 1.5
    PARAM_B = 0.75
    EPSILON = 0.25
    import numpy as np
    from gensim.summarization.bm25 import BM25
    frequency, texts = counter(advice)

    okapi_bm25 = BM25(texts)
    average_idf = sum(float(val) for val in okapi_bm25.idf.values()) / len(okapi_bm25.idf)

    bm25_dict = {}
    bm25_list = []
    bm25_jsd_dict = {}
    bm25_jsd_list = []
    calculation = Calc()
    # 例えば入力が「面接」だった時のトピック値としておく
    equation_lda_value = np.array(
                lda_value(["面接"], texts)['topic']
            )
    for index, text in enumerate(texts):
        bm25_dict[index] = {}
        bm25_jsd_dict[index] = {}
        report = advice[index]
        jsd = calculation.jsd(equation_lda_value, report["topic"])
        for word in text:
            idf = okapi_bm25.idf[word] if okapi_bm25.idf[word] >= 0 else EPSILON * average_idf

            score = 0
            score += (idf * okapi_bm25.f[index][word] * (PARAM_K1 + 1)
                      / (okapi_bm25.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * len(text) / okapi_bm25.avgdl)))
            bm25_dict[index][word] = score
            bm25_list.append(score)

            score_jsd = 0
            score_jsd += (idf * okapi_bm25.f[index][word] * (PARAM_K1 + 1)
                      / (okapi_bm25.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * jsd / okapi_bm25.avgdl)))
            bm25_jsd_dict[index][word] = score_jsd
            bm25_jsd_list.append(score_jsd)

    bm25_norm = dict_norm(bm25_dict, bm25_list)
    bm25_jsd_norm = dict_norm(bm25_jsd_dict, bm25_jsd_list)

    for key in advice:
        current_bm25 = bm25_norm[key]
        current_advice = advice[key]
        current_advice["bm25"] = current_bm25
        current_advice["bm25_sum"] = sum(flatten(current_bm25))
        current_advice["bm25_average"] = sum(flatten(current_bm25)) / (len(flatten(current_bm25)) + 0.001)

        current_bm25_jsd = bm25_jsd_norm[key]
        current_advice["bm25_jsd"] = current_bm25_jsd
        current_advice["bm25_jsd_sum"] = sum(flatten(current_bm25_jsd))
        current_advice["bm25_jsd_average"] = sum(flatten(current_bm25_jsd)) / (len(flatten(current_bm25_jsd)) + 0.001)

    return advice

if __name__ == '__main__':
    advice = load_pickle("../advice_2.pickle")
    advice = gensim_tfidf(advice)
    advice = gensim_bm25(advice)

    dump_pickle(advice, 'advice_2_tfidf.pickle')

