# -*- coding: utf-8 -*-
#!/usr/local/pyenv/shims/python

import gensim
from gensim.models import word2vec
import sys
import collections
import numpy as np
import pickle
import compTypeList
import math
from parse import parser_mecab
from parse import is_noun
from replace import change_word
from replace import decode_word
from calc import Calc
from addTopic import lda_value

#定数の宣言
similaryty = 0.50 # 類似度を設定する
INPUTWEIGHT = 1.0 # 入力文字の重み（仮想的な類似度）
PRIORITYRATE = 5 # 重要単語を選択した時に付加する類似語の類似度の倍率
LOWPRIORITYRATE = 0.5 # 非重要単語を選択した時に付加する類似語の類似度の倍率
WRITE = False # 入力内容を書き込むか否か Trueなら書き込み、Falseなら書き込まない
WEIGHTING = True # 入力文字のから重要単語を選択する場合はTrue,しない場合はFalse
TYPE = False
############

model = word2vec.Word2Vec.load(sys.argv[1])

# bizreachのモデルを利用する場合は以下のコメントアウトを削除する
#from gensim.models import KeyedVectors
#MODEL_FILENAME = "/var/www/cgi-bin/word-similarity-python/model/bizreach.model"
#model = KeyedVectors.load_word2vec_format(MODEL_FILENAME, binary=True)
# LDAによるトピック分類を利用した推薦のためのモデル読み込み

def min_max(x, min_x, max_x, axis=None):
    result = (x-min_x)/(max_x-min_x)
    return result

def list_checked(report_company_type, input_company_type):
    if report_company_type not in input_company_type or input_company_type == None:
        rate = 1
    else:
        rate = 1.2
    return rate

def cos_norm(cosSimilar):
    calc = Calc()
    list_cos = [cos for cos in cosSimilar.values()]
    for key in cosSimilar:
        current_cos = cosSimilar[key]
        cosSimilar[key] = calc.normalization(current_cos, list_cos)
    return cosSimilar

def neighbor_word(posi, nega=[], n=300, inputText = None):
    tmpWordCheck = ''
    count = 0

    results = []
    inputVectorSum = 0 # 入力文字のベクトルの和を格納
    inputVectorLength = 0 # 入力文字のベクトル長を格納
    resultWord = [] # 入力文字の中でword2vecによって学習されている単語を格納する
    posi = sorted(list(set(posi)), key=posi.index)
    for inputWord in posi:
        inputWord = change_word(inputWord)
        try:
            model.most_similar(positive = inputWord, negative = nega, topn = n)
            resultWord.append(inputWord)
        except  Exception as e:
            continue
        results.append((inputWord, INPUTWEIGHT))
    posi = resultWord
    for index, po in enumerate(posi): # 入力文字から類似語を出力
        try:
            result = model.most_similar(positive = po, negative = nega, topn = n)
            tmpWordCheck += '1,' + po + ','
            for r in result:
                if r[1] > similaryty:
                    if WEIGHTING == True and not ALGORITHMTYPE == 0:
                        results.append(r)
                    else:
                        if index == int(weightingFlag): # 入力の中で重要であると利用者が判断した単語の類似語の類似度を少し増やす
                            results.append((r[0], r[1] * PRIORITYRATE))
                        else:
                            results.append((r[0], r[1] * LOWPRIORITYRATE))
                else:
                    break;
            # 入力のベクトルの和
            inputVectorSum += model[po]
        except  Exception as e:
            tmpWordCheck += '0,' + po + ','
        count += 1
    inputVectorLength = np.linalg.norm(inputVectorSum)

    words = {'positive': posi, 'negative': nega}
    # adDictsのpickleをロードする
    with open('/var/www/cgi-bin/word-similarity-python/tfidf/advice_10_tfidf.pickle', 'rb') as f: # トピック分類の情報を付加したデータをpickleでロード
        adDicts = pickle.load(f)
    rateCount = []
    topicDic = {} # 入力と文書ごとのトピック積和を格納
    cosSimilar = {} # 入力と文書ごとのコサイン類似度を格納
    reportNoType = {} # 報告書Noと業種の辞書
    reportNoShokushu = {} # 報告書Noと職種の辞書
    wordDictionary = {} # 報告書ごとの類似単語辞書
    wordCount = {} # 類似単語の出現回数
    ldaDictionary = {} # 報告書ごとに入力とldaのtopic値を計算する
    equation_lda_value = np.array(lda_value(equation, [posi])['topic']) # 入力値にLDAによるtopic値を付与する
    for index in adDicts:
        wordDictionary[adDicts[index]["reportNo"]] = {}

    for kensaku in results:
        wordCount[kensaku[0]] = 0
        if not is_noun(kensaku[0]):
            continue
        for index in adDicts:
            if len(adDicts[index]['advice_divide_mecab']) < 10:
                continue
            if adDicts[index]['advice'] == '':
                continue
            #if adDicts[index]['advice_divide_mecab_space'].find(kensaku[0]) == -1: # adviceに類似度の高い単語が含まれている場合
            if kensaku[0] not in adDicts[index]['advice_divide_mecab']: # adviceに類似度の高い単語が含まれている場合
                continue
            report_no = adDicts[index]["reportNo"]
            wordDictionary[report_no].update({decode_word(kensaku[0]): kensaku[1]})
            if kensaku[0] in adDicts[index]['tfidf']:
                rateCount.append([report_no, adDicts[index]["companyName"], adDicts[index]['tfidf'][kensaku[0]] * kensaku[1]])
            else:
                rateCount.append([report_no, adDicts[index]["companyName"], kensaku[1]]) # 類似度を用いて推薦機能を実装するための配列
            reportNoType[report_no] = adDicts[index]["companyType"]
            reportNoShokushu[report_no] = adDicts[index]["companyShokushu"]
            cosSimilar[report_no] = np.dot(adDicts[index]['vectorSum'], inputVectorSum) / (adDicts[index]['vectorLength'] * inputVectorLength) # 入力の文章と各文書ごとにコサイン類似度を計算
            ldaDictionary[report_no] = sum(equation_lda_value * np.array(adDicts[index]['topic']))
            wordCount[kensaku[0]] += 1

    # 内積の計算でコサイン類似度がマイナスになることがあったので、正規化した
    cosSimilar = cos_norm(cosSimilar)

    reportDict = {} # 類似語を含むアドバイスの類似度をreport_no毎に足し算する
    # 同じ企業名で類似度を合計する
    fno1Comp = collections.Counter([comp[0] for comp in rateCount])
    rateCountNp = np.array(rateCount)

    compRecommendDic = {}
    t = 0
    reportNp = rateCountNp[:, [0]].reshape(-1,)
    rateCountNp = rateCountNp[:, [1, 2]]

    for comp_no in fno1Comp:
        typeRate = list_checked(reportNoType[comp_no], company_type_name)
        shokushuRate = list_checked(reportNoShokushu[comp_no], company_shokushu_name)
        similarSum = rateCountNp[np.where(reportNp == str(comp_no))]
        simSum = np.sum(similarSum[:,1].reshape(-1,).astype(np.float64))
        simLog = 0.0001 if math.log(simSum, 10) < 0 else math.log(simSum, 10)
        if ALGORITHMTYPE == 0:
            # type0: 類似語の合計 * 業種（メタ情報） * コサイン類似度
            compRecommendDic[comp_no] = simSum * typeRate * cosSimilar[comp_no]
        elif ALGORITHMTYPE == 1:
            # type1: log(類似語の合計) * 業種（メタ情報） * 職種（メタ情報）* コサイン類似度
            #compRecommendDic[comp_no] = simLog * typeRate * shokushuRate * cosSimilar[comp_no]

            # 2018-06-07 type1:                    log(sum(similarity)) + コサイン類似度 *（メタ情報）
            #compRecommendDic[comp_no] = simLog + cosSimilar[comp_no] * (typeRate * shokushuRate)

            # 2018-06-12 type1:                    log(sum(similarity)) + コサイン類似度 *（メタ情報）
            compRecommendDic[comp_no] = simLog + ldaDictionary[comp_no] * (typeRate * shokushuRate)
        elif ALGORITHMTYPE == 2:
            # type2: log(類似語の合計) + 業種（メタ情報） + コサイン類似度
            compRecommendDic[comp_no] = simLog + typeRate + cosSimilar[comp_no]


    advice_json = {}
    for index, primaryComp in enumerate(sorted(compRecommendDic.items(), key=lambda x: x[1], reverse=True)[:100]):
        ranking = index + 1
        advice_json[str(ranking)] = {
                'report_no': primaryComp[0],
                'recommend_level': str(primaryComp[1]),
                'words': wordDictionary[primaryComp[0]],
                'cos': cosSimilar[primaryComp[0]].astype(np.float64),
                'lda': ldaDictionary[primaryComp[0]].astype(np.float64),
                }
    # ワードクラウド用に類似単語の出現回数を取得してみる
    [wordCount.pop(w[0]) for w in list(wordCount.items()) if w[1] == 0]
    advice_json['word_count'] = sorted(wordCount.items(), key=lambda x:x[1], reverse=True)
    advice_json['company_type'] = company_type_name
    advice_json['company_shokushu'] = company_shokushu_name
    return json_dump(advice_json)

def json_dump(dictionary):
    import json
    return json.dumps(dictionary, sort_keys=True)

def calc(equation):
    words = parser_mecab(equation)
    return neighbor_word(words, inputText=equation)

if __name__=="__main__":
    ALGORITHMTYPE = 1
    equation = sys.argv[2]
    company_type_name = sys.argv[3].split()
    company_shokushu_name = sys.argv[4].split()
    print(calc(equation))


