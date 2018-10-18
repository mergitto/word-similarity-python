from gensim import corpora
from collections import defaultdict
import pickle
import math

def load_pickle():
    with open('../advice_10.pickle', 'rb') as f:
        advice = pickle.load(f)
    return advice

def dump_pickle(data):
    with open('advice_10_tfidf.pickle', 'wb') as f:
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

def greater_than_zero(array):
    return len(array) > 0

def gensim_tfidf(advice):
    frequency, texts = counter(advice)

    # id:単語　の形
    dictionary = corpora.Dictionary(texts)
    # corpus[0]で0番目の文書のbag-of-wordを取得できる
    corpus = bag_of_word(dictionary, texts)
    corpus_tfidf = load_tfidf_model(corpus)

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

        if greater_than_zero(corpus_of_each_text):
            tfidf_average = tfidf_vector_sum / len(corpus_of_each_text)
            current_advice['tfidf_average'] = tfidf_average
        else:
            current_advice['tfidf_average'] = 0

    return advice

if __name__ == '__main__':
    advice = load_pickle()
    advice = gensim_tfidf(advice)

    dump_pickle(advice)

