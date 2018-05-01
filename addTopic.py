# -*- coding: utf-8 -*-

import gensim
from gensim import corpora
from gensim.models import word2vec
import pandas as pd
from collections import defaultdict
from pprint import pprint
import urllib.request
from janome.tokenizer import Tokenizer
from natto import MeCab
import pickle
import re
import sys
import numpy as np
from const import PATH


# 分かち書きの文章を利用して学習を行う
def trainLda():
    # トピック分類のための学習用文章データ
    rows = pd.read_table(PATH['WAKACHI'], header=None)

    print(rows[0])
    print(type(rows))
    t = Tokenizer()
    documents = [] # 文章の名詞のみのリストを作成

    for row in rows[0]:
        # ワードのベクトル
        word_vector = []

        if len(row) < 20: # 文章のサイズが20以下の場合は学習対象としない
            continue;
        else:
            tokens = t.tokenize(row)

        # 名詞のみの単語に変更する
        for token in tokens:
            if token.part_of_speech[:2] == '名詞':
                word_vector += [token.base_form]
        documents.append(word_vector)

    # ストップワードの読み込み
    f = urllib.request.urlopen('http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt')
    sw = [line.decode('utf-8').strip() for line in f] # 読み込んだurlから文章を読み込み
    sw = set(sw)

    # ストップワードの除去
    texts = [[word for word in document if word not in sw] for document in documents]

    # 出現頻度が１回の単語を削除
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]


    dictionary = corpora.Dictionary(texts)
    dictionary.save('./ldaModel/lda_%s.txt' % TOPICNUM)

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('./ldaModel/lda_%s.mm' % TOPICNUM, corpus)

    #LDAモデルによる学習
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=TOPICNUM, id2word=dictionary)
    lda.save('./ldaModel/lda_%s.model' % TOPICNUM)  # 保存


# 訓練したデータをロードしてトピック分類を行う
def loadLda(text=None):
    dictionary = corpora.Dictionary.load('./ldaModel/lda_%s.txt' % TOPICNUM)
    corpus = corpora.MmCorpus('./ldaModel/lda_%s.mm' % TOPICNUM)
    lda = gensim.models.ldamodel.LdaModel.load('./ldaModel/lda_%s.model' % TOPICNUM)
    # 学習により得たトピック
    #pprint(lda.show_topics(num_topics=TOPICNUM))
    # ldaによるトピックをcsvで出力
    pd.DataFrame(lda.show_topics(num_topics=TOPICNUM)).to_csv("./ldaModel/topic_%s.csv" % TOPICNUM, header=None, index=None)

    test_words = ""
    for n in mc.parse(text, as_nodes=True):
        node = n.feature.split(',');
        if node[0] != '助詞' and node[0] != '助動詞' and node[0] != '記号' and node[1] != '数':
            if node[0] == '動詞':
                test_words += node[6]
            else:
                test_words += n.surface
            test_words += " "
    # テスト用で適当な文章を作成し、どのトピックに当たるかを出力させてみる
    test_documents = [test_words]
    test_texts = [[word for word in document.split()] for document in test_documents]
    test_corpus = [dictionary.doc2bow(text) for text in test_texts]

    # 文書に付いているトピックを計算する
    for topics_per_document in lda[test_corpus]:
        topicDict = {}
        topicList = []
        topicCount = len(topics_per_document) #トピック数のカウント、繰り返し制御用
        topicList = [0 for i in range(TOPICNUM)] # 学習時の全トピック数によるリストの初期化
        for x in range(topicCount):
            topicList[topics_per_document[x][0]] = topics_per_document[x][1] # トピックに属していたもののみリストの修正
        topicDict['topic'] = topicList

    # ストップワードの除去しつつベクトルの和を計算
    f = urllib.request.urlopen('http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt')
    sw = [line.decode('utf-8').strip() for line in f] # 読み込んだurlから文章を読み込み
    sw = [ss for ss in sw if not ss==u''] # 空白を削除
    f.close()
    vectorSum = 0 # 文書の単語ごとのベクトルの和を格納する
    for word in test_texts[0]:
        if not word in sw:
            vectorSum += model[word]
    topicDict['vectorSum'] = vectorSum
    topicDict['vectorLength'] = np.linalg.norm(vectorSum)

    return topicDict


mc = MeCab('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
TOPICNUM = 10 # 学習したトピック数
train = False # True = 学習を行う False = 訓練したデータをロードして結果を表示する

model   = word2vec.Word2Vec.load(sys.argv[1]) # 文書ごとにベクトルの和を求めるためにword2vecモジュールを読み込み
if __name__ == '__main__':
    if train:
        trainLda()
    else:
        with open('./advice.pickle', 'rb') as f:
            sugAd = pickle.load(f)
        for index in sugAd:
            if sugAd[index]['advice'] == None:
                sugAd[index].update({'topic': [0 for i in range(TOPICNUM)]})
                print('報告書No:', str(sugAd[index]['reportNo']), 'トピックリスト作成中:',round(index/len(sugAd), 4),'%')
            else:
                value = sugAd[index]
                # アドバイスデータの加工
                document = value['advice'] # アドバイスだけ避難
                document = re.sub("\<.+?\>", "", document)
                document = re.sub("\[.+?\]", "", document)
                document = re.sub("\（|(.+?\）|)", "", document)
                topic = loadLda(document)
                sugAd[index].update({'topic':topic['topic']}) # topicListを加えたデータに更新
                sugAd[index].update({'vectorSum':topic['vectorSum']}) # 文書のベクトルの和を加えたデータに更新
                sugAd[index].update({'vectorLength':topic['vectorLength']}) # 文書のベクトルの和を加えたデータに更新
                print('報告書No:', str(sugAd[index]['reportNo']), 'トピックリスト作成中:',round(index/len(sugAd), 4),'%')

        with open('./advice_%s.pickle' % TOPICNUM, 'wb') as f:
            pickle.dump(sugAd, f)

