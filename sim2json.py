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
from const import *

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

def is_exist_input_word(inputWord, model):
    try:
        model.most_similar(positive = inputWord, negative = [], topn = NEIGHBOR_WORDS)
        return True
    except  Exception as e:
        return False

def high_similar_words(result, results):
    for r in result:
        if r[1] > SIMILARYTY_LIMIT_RATE:
            results.append(r)
        else:
            continue;
    return results

def get_similar_words(inputWord):
    results = []
    inputVectorSum = 0 # 入力文字のベクトルの和を格納
    for index, word in enumerate(inputWord): # 入力文字から類似語を出力
        try:
            if is_exist_input_word(word, model): results.append((word, INPUTWEIGHT))
            result = model.most_similar(positive = word, negative = [], topn = NEIGHBOR_WORDS)
            results = high_similar_words(result, results)
            # 入力のベクトルの和
            inputVectorSum += model[word]
        except  Exception as e:
            pass
    return results, inputVectorSum

def load_reports():
    with open(PATH["REPORTS_PICKELE"], 'rb') as f: # トピック分類の情報を付加したデータをpickleでロード
        adDicts = pickle.load(f)
    return adDicts

def is_not_match_report(company_type, company_shokushu):
    if det_check == "1":
        if company_type not in company_type_name and company_shokushu not in company_shokushu_name:
            return True
    return False

def neighbor_word(posi, nega=[], n=NEIGHBOR_WORDS, inputText = None):
    posi = sorted(list(set(posi)), key=posi.index)
    results, inputVectorSum = get_similar_words(posi)
    inputVectorLength = np.linalg.norm(inputVectorSum) # 入力文字のベクトル長を格納

    adDicts = load_reports()

    rateCount = []
    cosSimilar = {} # 入力と文書ごとのコサイン類似度を格納
    reportNoType = {} # 報告書Noと業種の辞書
    reportNoShokushu = {} # 報告書Noと職種の辞書
    wordDictionary = {} # 報告書ごとの類似単語辞書
    wordCount = {} # 類似単語の出現回数
    ldaDictionary = {} # 報告書ごとに入力とldaのtopic値を計算する
    jsdDictionary = {} # 報告書ごとに入力とldaのtopic値を活用してjsd値を計算する
    lda = {}
    equation_lda_value = np.array(lda_value(equation, [posi])['topic']) # 入力値にLDAによるtopic値を付与する
    for index in adDicts:
        wordDictionary[adDicts[index]["reportNo"]] = {}
    calc = Calc()

    for kensaku in results:
        wordCount[kensaku[0]] = 0
        if not is_noun(kensaku[0]): continue
        for index in adDicts:
            report = adDicts[index]
            if len(report['advice_divide_mecab']) < LOWEST_WORD_LENGTH: continue
            if report['advice'] == '': continue
            report_no = report["reportNo"]
            ldaDictionary[report_no] = sum(equation_lda_value * np.array(report['topic']))
            jsdDictionary[report_no] = calc.jsd(equation_lda_value, np.array(report['topic']))
            cosSimilar[report_no] = np.dot(report['vectorSum'], inputVectorSum) / (report['vectorLength'] * inputVectorLength)
            if kensaku[0] not in report['advice_divide_mecab']: continue
            if is_not_match_report(report["companyType"], report["companyShokushu"]): continue
            wordDictionary[report_no].update({decode_word(kensaku[0]): kensaku[1]})
            if kensaku[0] in report['tfidf']:
                rateCount.append([report_no, report["companyName"], report['tfidf'][kensaku[0]] * kensaku[1]])
            else:
                rateCount.append([report_no, report["companyName"], kensaku[1]])
            reportNoType[report_no] = report["companyType"]
            reportNoShokushu[report_no] = report["companyShokushu"]
            lda[report_no] = [report['topic'][0], report['topic'][1]]
            wordCount[kensaku[0]] += 1

    cosSimilar = normalization(cosSimilar)
    ldaDictionary = normalization(ldaDictionary)

    # jsdは非類似度が高いほど値が大きくなるので、値が大きいほど類似度が高くなるように修正
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

    for comp_no in fno1Comp:
        typeRate = list_checked(reportNoType[comp_no], company_type_name)
        shokushuRate = list_checked(reportNoShokushu[comp_no], company_shokushu_name)
        similarSum = rateCountNp[np.where(reportNp == str(comp_no))]
        if len(similarSum) == 0:
            similarSum = 0
        else:
            simSum = np.sum(similarSum[:,1].reshape(-1,).astype(np.float64))
        simLog = 0.0001 if math.log(simSum, 2) <= 0 else math.log(simSum, 10)
        simLog = simLog * 1.2
        compRecommendDic[comp_no] = simLog + cosSimilar[comp_no] * jsdDictionary[comp_no] * (typeRate * shokushuRate)


    advice_json = {}
    for index, primaryComp in enumerate(sorted(compRecommendDic.items(), key=lambda x: x[1], reverse=True)[:DISPLAY_REPORTS_NUM]):
        ranking = index + 1
        advice_json[str(ranking)] = {
                'report_no': primaryComp[0],
                'recommend_level': str(round(primaryComp[1], DECIMAL_POINT)),
                'words': wordDictionary[primaryComp[0]],
                'cos': round(cosSimilar[primaryComp[0]].astype(np.float64), DECIMAL_POINT),
                'lda1': lda[primaryComp[0]][0],
                'lda2': lda[primaryComp[0]][1],
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


model = word2vec.Word2Vec.load(sys.argv[1])

if __name__=="__main__":
    equation = change_word(sys.argv[2])
    company_type_name = sys.argv[3].split()
    company_shokushu_name = sys.argv[4].split()
    det_check = sys.argv[5]
    similarReports = calc(equation)
    print(similarReports)


