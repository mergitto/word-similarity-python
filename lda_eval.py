# 使用方法
# python lda_value.py
# coherenceとperplexityを求める処理

from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.models import word2vec
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import os
import numpy as np
from const import PATH

ROOTPATH = os.path.dirname(os.path.abspath(__file__))

sentences = [s for s in word2vec.LineSentence(PATH['WAKACHI']) if len(s) >= 2]
dic = Dictionary(sentences)
corpus = [dic.doc2bow(s) for s in sentences]

# c_v, c_uci, c_npmi用に、学習に用いたコーパスとは別のコーパスからtextを用意
dictfile='./ldaModel/lda_2.txt'
dictionary = Dictionary.load(dictfile) # 辞書読み込み
texts = []
another_corpus = MmCorpus("./ldaModel/lda_2.mm")
for doc in another_corpus:
    text = []
    for word in doc:
        for i in range(int(word[1])):
            text.append(dictionary[word[0]])
    texts.append(text)

for i in range(1, 11):
    lda = LdaModel(corpus = corpus, id2word = dic, num_topics = i, alpha = 0.01, random_state = 1)

    cm = CoherenceModel(model = lda, texts  = texts, dictionary=dictionary, coherence = 'c_v')
    coherence = cm.get_coherence()

    perwordbound = lda.log_perplexity(corpus)
    perplexity = np.exp2(-perwordbound)

    print(f"num_topics = {i}, coherence = {coherence}, perplexity = {perplexity}")

