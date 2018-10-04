# -*- coding: utf-8 -*-
#!/usr/local/pyenv/shims/python

import gensim
from gensim.models import word2vec
import sys
import collections
import numpy as np
import pickle
import math
from parse import parser_mecab
from parse import is_noun
from replace import change_word
from replace import decode_word
from calc import Calc
from addTopic import lda_value
from pprint import pprint

#定数の宣言
similaryty = 0.50 # 類似度を設定する
INPUTWEIGHT = 1.0 # 入力文字の重み（仮想的な類似度）
PRIORITYRATE = 5 # 重要単語を選択した時に付加する類似語の類似度の倍率
LOWPRIORITYRATE = 0.5 # 非重要単語を選択した時に付加する類似語の類似度の倍率
WRITE = False # 入力内容を書き込むか否か Trueなら書き込み、Falseなら書き込まない
WEIGHTING = True # 入力文字のから重要単語を選択する場合はTrue,しない場合はFalse
TYPE = False
############


def min_max(x, min_x, max_x, axis=None):
    result = (x-min_x)/(max_x-min_x)
    return result

def list_checked(report_company_type, input_company_type):
    if report_company_type not in input_company_type or input_company_type == None:
        rate = 1
    else:
        rate = 1.2
    return rate

def normalization(cosSimilar):
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
                    if WEIGHTING == True:
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
    #with open('./tfidf/advice_10_tfidf.pickle', 'rb') as f: # トピック分類の情報を付加したデータをpickleでロード
    with open('./tfidf/advice_classification_2.pickle', 'rb') as f: # トピック分類の情報を付加したデータをpickleでロード
        adDicts = pickle.load(f)
    rateCount = []
    topicDic = {} # 入力と文書ごとのトピック積和を格納
    cosSimilar = {} # 入力と文書ごとのコサイン類似度を格納
    reportNoType = {} # 報告書Noと業種の辞書
    reportNoShokushu = {} # 報告書Noと職種の辞書
    wordDictionary = {} # 報告書ごとの類似単語辞書
    wordCount = {} # 類似単語の出現回数
    ldaDictionary = {} # 報告書ごとに入力とldaのtopic値を計算する
    jsdDictionary = {} # 報告書ごとに入力とldaのtopic値を活用してjsd値を計算する
    report_no_list = [] # 類似単語が無かった時に推薦どを低くするやつ
    equation_lda_value = np.array(lda_value(equation, [posi])['topic']) # 入力値にLDAによるtopic値を付与する
    highlowDictionary = {} # 検証用の高評価・低評価判定用の辞書
    for index in adDicts:
        wordDictionary[adDicts[index]["reportNo"]] = {}
        report_no_list.append(adDicts[index]["reportNo"])
    calc = Calc()

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
            report_no = adDicts[index]["reportNo"]
            ldaDictionary[report_no] = sum(equation_lda_value * np.array(adDicts[index]['topic']))
            jsdDictionary[report_no] = calc.jsd(equation_lda_value, np.array(adDicts[index]['topic']))
            highlowDictionary[report_no] = adDicts[index]['evaluation']
            cosSimilar[report_no] = np.dot(adDicts[index]['vectorSum'], inputVectorSum) / (adDicts[index]['vectorLength'] * inputVectorLength) # 入力の文章と各文書ごとにコサイン類似度を計算
            if kensaku[0] not in adDicts[index]['advice_divide_mecab']: # adviceに類似度の高い単語が含まれている場合
                continue
            wordDictionary[report_no].update({decode_word(kensaku[0]): kensaku[1]})
            if kensaku[0] in adDicts[index]['tfidf']:
                rateCount.append([report_no, adDicts[index]["companyName"], adDicts[index]['tfidf'][kensaku[0]] * kensaku[1]])
            else:
                rateCount.append([report_no, adDicts[index]["companyName"], kensaku[1]]) # 類似度を用いて推薦機能を実装するための配列
            reportNoType[report_no] = adDicts[index]["companyType"]
            reportNoShokushu[report_no] = adDicts[index]["companyShokushu"]
            wordCount[kensaku[0]] += 1

    # 内積の計算でコサイン類似度がマイナスになることがあったので、正規化した
    cosSimilar = normalization(cosSimilar)
    ldaDictionary = normalization(ldaDictionary)
    jsdDictionary = normalization(jsdDictionary)
    jsdDictionary = calc.value_reverse(jsdDictionary)

    reportDict = {} # 類似語を含むアドバイスの類似度をreport_no毎に足し算する
    # 同じ企業名で類似度を合計する
    fno1Comp = collections.Counter([comp[0] for comp in rateCount])
    rateCountNp = np.array(rateCount)

    compRecommendDic = {}
    t = 0
    reportNp = rateCountNp[:, [0]].reshape(-1,)
    rateCountNp = rateCountNp[:, [1, 2]]

    for comp_no in report_no_list:#fno1Comp:
        #typeRate = list_checked(reportNoType[comp_no], company_type_name)
        #shokushuRate = list_checked(reportNoShokushu[comp_no], company_shokushu_name)
        similarSum = rateCountNp[np.where(reportNp == str(comp_no))]
        if len(similarSum) == 0:
            similarSum = 0
        else:
            simSum = np.sum(similarSum[:,1].reshape(-1,).astype(np.float64))
        simLog = 0.0001 if math.log(simSum, 2) <= 0 else math.log(simSum, 10)
        compRecommendDic[comp_no] = simLog + cosSimilar[comp_no] * jsdDictionary[comp_no]


    advice_json = {}
    for index, primaryComp in enumerate(sorted(compRecommendDic.items(), key=lambda x: x[1], reverse=True)[:20]):
        ranking = index + 1
        advice_json[str(ranking)] = {
                'report_no': primaryComp[0],
                'recommend_level': str(round(primaryComp[1], 3)),
                #'words': wordDictionary[primaryComp[0]],
                #'cos': round(cosSimilar[primaryComp[0]].astype(np.float64), 3),
                #'lda': round(ldaDictionary[primaryComp[0]].astype(np.float64), 3),
                'evaluation': highlowDictionary[primaryComp[0]],
                }
    # ワードクラウド用に類似単語の出現回数を取得してみる
    [wordCount.pop(w[0]) for w in list(wordCount.items()) if w[1] == 0]
    #advice_json['word_count'] = sorted(wordCount.items(), key=lambda x:x[1], reverse=True)
    #advice_json['company_type'] = company_type_name
    #advice_json['company_shokushu'] = company_shokushu_name
    write_txt(advice_json)
    return json_dump(advice_json)

def write_txt(dictionary):
    high_count = 0
    MODEL_FILE_NAME = "cpf_sum"
    VERIFICATION_FILE = "./verification/%s_recommend.txt" % MODEL_FILE_NAME
    with open(VERIFICATION_FILE, mode="a") as f:
        f.write("[FORMAT]\n")
        f.write("rank(high_report_number):{value}\n")
        for index in dictionary:
            value = dictionary[index]
            if value["evaluation"] == 'high': high_count += 1
            report_val = str(index)+"(%s)"% high_count+":"+str(value)
            f.write(report_val+"\n")
        f.write("\n")

def json_dump(dictionary):
    import json
    return json.dumps(dictionary, sort_keys=True)

def calc(equation):
    words = parser_mecab(equation)
    return neighbor_word(words, inputText=equation)


model = word2vec.Word2Vec.load(sys.argv[1])
# 検証用のファイル

if __name__=="__main__":
    equation = sys.argv[2]
    company_type_name = sys.argv[3].split()
    company_shokushu_name = sys.argv[4].split()
    similarReports = calc(equation)
    pprint(similarReports, width=80)


