from gensim import corpora
from collections import defaultdict
import pickle
import math

with open('../advice_10.pickle', 'rb') as f:
    advice = pickle.load(f)

def new_idf(docfreq, totaldocs, log_base=2.0, add=1.0):
    return add + math.log(1.0 * totaldocs / docfreq, log_base)

def gensim_tfidf(advice):
    texts = []

    frequency = defaultdict(int)
    for reportNo in advice:
        for token in advice[reportNo]['advice_divide_mecab']:
            frequency[token] += 1
        texts.append(advice[reportNo]['advice_divide_mecab'])

    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    # id:単語　の形
    dictionary = corpora.Dictionary(texts)
    # corpus[0]で0番目の文書のbag-of-wordを取得できる
    corpus = [dictionary.doc2bow(text) for text in texts]

    from gensim import models
    tfidf = models.TfidfModel(corpus)#, wglobal=new_idf)#, normalize=False)
    # corpus_tfidf[0]で0番目の文書の単語のcorpusのtfidfを表示
    corpus_tfidf = tfidf[corpus]
    for index, corpus_of_each_text in enumerate(corpus):
        if index % 10 == 0:
            print("進捗度: ", str(round((index+1) / len(corpus) * 100, 2)), '%')

        advice[index]['tfidf'] = {}
        for one_corpus in corpus_of_each_text:
            tfidf_tuple = [ct for ct in corpus_tfidf[index] if ct[0] == one_corpus[0]][0]
            advice[index]['tfidf'][dictionary[tfidf_tuple[0]]] = tfidf_tuple[1]

    return advice

if __name__ == '__main__':
    advice = gensim_tfidf(advice)

    with open('advice_10_tfidf.pickle', 'wb') as f:
        pickle.dump(advice, f)


