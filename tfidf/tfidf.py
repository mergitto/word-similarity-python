from gensim import corpora
from collections import defaultdict
import pickle
import math

def load_pickle():
    with open('../advice_10.pickle', 'rb') as f:
        advice = pickle.load(f)
    return advice

def load_tfidf_model(corpus):
    from gensim import models
    return models.TfidfModel(corpus)

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

def gensim_tfidf(advice):
    frequency, texts = counter(advice)

    # id:単語　の形
    dictionary = corpora.Dictionary(texts)
    # corpus[0]で0番目の文書のbag-of-wordを取得できる
    corpus = bag_of_word(dictionary, texts)

    tfidf = load_tfidf_model(corpus)
    # corpus_tfidf[0]で0番目の文書の単語のcorpusのtfidfを表示
    corpus_tfidf = tfidf[corpus]
    for index, corpus_of_each_text in enumerate(corpus):
        if index % 10 == 0:
            print("進捗度: ", str(round((index+1) / len(corpus) * 100, 2)), '%')

        advice[index]['tfidf'] = {}
        tfidf_vector_sum = 0
        for one_corpus in corpus_of_each_text:
            tfidf_tuple = [ct for ct in corpus_tfidf[index] if ct[0] == one_corpus[0]][0]
            advice[index]['tfidf'][dictionary[tfidf_tuple[0]]] = tfidf_tuple[1]
            tfidf_vector_sum += tfidf_tuple[1]

        if len(corpus_of_each_text) > 0:
            tfidf_average = tfidf_vector_sum / len(corpus_of_each_text)
            advice[index]['tfidf_average'] = tfidf_average
        else:
            advice[index]['tfidf_average'] = 0

    return advice

if __name__ == '__main__':
    advice = load_pickle()
    advice = gensim_tfidf(advice)

    with open('advice_10_tfidf.pickle', 'wb') as f:
        pickle.dump(advice, f)


