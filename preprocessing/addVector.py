# -*- coding: utf-8 -*-

import gensim
from gensim import corpora
from gensim.models.word2vec import Word2Vec
import pandas as pd
import urllib.request
from natto import MeCab
import pickle
import re
import numpy as np
from tqdm import tqdm
import os, sys
sys.path.append("../")
from const import WORD2VECMODELFILE
from parse import parser_mecab

model = Word2Vec.load(WORD2VECMODELFILE)

def get_stop_word():
    # ストップワードの取得
    stop_word = urllib.request.urlopen(
            'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt')
    sw = [line.decode('utf-8').strip() for line in stop_word] # 読み込んだurlから文章を読み込み
    sw = [ss for ss in sw if not ss==u''] # 空白を削除
    stop_word.close()
    return sw

def vector_sum(parse_text):
    vectorSum = 0 # 文書の単語ごとのベクトルの和を格納する
    stop_word = get_stop_word()
    for word in parse_text:
        if word in stop_word: continue
        try:
            vectorSum += model[word]
        except:
            pass
    return vectorSum

def add_vector(text=None):
    parse_text = [parser_mecab(text)]
    results = {}
    vectorSum = vector_sum(parse_text[0])
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
    with open('../advice.pickle', 'rb') as f:
        reports = pickle.load(f)

    for index in tqdm(reports):
        report = reports[index]
        document = document_norm(report['advice'])
        vector_values = add_vector(document)
        reports[index].update({
                    'vectorSum': vector_values['vectorSum'],
                    'vectorLength': vector_values['vectorLength']
                })

    with open('../advice_add_vector.pickle', 'wb') as f:
        pickle.dump(reports, f)

