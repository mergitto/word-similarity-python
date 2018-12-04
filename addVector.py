# -*- coding: utf-8 -*-

import gensim
from gensim import corpora
from gensim.models.word2vec import Word2Vec
import pandas as pd
import urllib.request
from natto import MeCab
import pickle
import re
import sys
import numpy as np
from const import WORD2VECMODELFILE
from parse import parser_mecab
import os

ROOTPATH = os.path.dirname(os.path.abspath(__file__))
model = Word2Vec.load(WORD2VECMODELFILE)

def add_vector(text=None):
    parse_text = [parser_mecab(text)]
    results = {}

    # ストップワードの除去しつつベクトルの和を計算
    f = urllib.request.urlopen('http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt')
    sw = [line.decode('utf-8').strip() for line in f] # 読み込んだurlから文章を読み込み
    sw = [ss for ss in sw if not ss==u''] # 空白を削除
    f.close()
    vectorSum = 0 # 文書の単語ごとのベクトルの和を格納する
    for word in parse_text[0]:
        if not word in sw:
            try:
                vectorSum += model[word]
            except:
                pass
    results['vectorSum'] = vectorSum
    results['vectorLength'] = np.linalg.norm(vectorSum)

    return results

# アドバイスデータの正規化とか記号とかの削除
def document_norm(document):
    document = re.sub("\<.+?\>", "", document)
    document = re.sub("\[.+?\]", "", document)
    document = re.sub("\（|(.+?\）|)", "", document)
    return document


if __name__ == '__main__':
    with open('./advice.pickle', 'rb') as f:
        sugAd = pickle.load(f)

    for index in sugAd:
        report = sugAd[index]
        document = document_norm(report['advice'])
        vector_values = add_vector(document)
        sugAd[index].update({'vectorSum': vector_values['vectorSum']}) # 文書のベクトルの和を加えたデータに更新
        sugAd[index].update({'vectorLength': vector_values['vectorLength']}) # 文書のベクトルの和を加えたデータに更新
        print('報告書No:', str(sugAd[index]['reportNo']), 'トピックリスト作成中:',round(index/len(sugAd) * 100, 1),'%')

    with open('./advice_add_vector.pickle', 'wb') as f:
        pickle.dump(sugAd, f)

