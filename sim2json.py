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

def normalization(dictionary):
    calc = Calc()
    values = [value for value in dictionary.values()]
    for key in dictionary:
        current_values = dictionary[key]
        dictionary[key] = calc.normalization(current_values, values)
    return dictionary

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
    for index, word in enumerate(inputWord): # 入力文字から類似語を出力
        try:
            if is_exist_input_word(word, model): results.append((word, INPUTWEIGHT))
            result = model.most_similar(positive = word, negative = [], topn = NEIGHBOR_WORDS)
            results = high_similar_words(result, results)
        except  Exception as e:
            pass
    return results

def load_reports():
    with open(PATH["REPORTS_PICKELE"], 'rb') as f: # トピック分類の情報を付加したデータをpickleでロード
        adDicts = pickle.load(f)
    return adDicts

def is_not_match_report(company_type, company_shokushu):
    if det_check == "1":
        if company_type not in company_type_name and company_shokushu not in company_shokushu_name:
            return True
    return False

def calcSimSum(similarSumary):
    if len(similarSumary) == 0:
        simSum = 0
    else:
        simSum = np.sum(similarSumary[:,1].reshape(-1,).astype(np.float64))
    return simSum

def calcSimLog(simSum):
    simLog = 0.0001 if math.log(simSum, 2) <= 0 else math.log(simSum, 10)
    return simLog * 1.2

def is_few_words(parse_text):
    if len(parse_text) < LOWEST_WORD_LENGTH:
        return True
    return False

def neighbor_word(posi, nega=[], n=NEIGHBOR_WORDS, inputText = None):
    posi = sorted(list(set(posi)), key=posi.index)
    results = get_similar_words(posi)

    rateCount = []
    reportNoType = {} # 報告書Noと業種の辞書
    reportNoShokushu = {} # 報告書Noと職種の辞書
    wordDictionary = {} # 報告書ごとの類似単語辞書
    wordCount = {} # 類似単語の出現回数
    jsdDictionary = {} # 報告書ごとに入力とldaのtopic値を活用してjsd値を計算する
    lda = {}
    equation_lda_value = np.array(lda_value(equation, [posi])['topic']) # 入力値にLDAによるtopic値を付与する

    adDicts = load_reports()
    for index in adDicts:
        wordDictionary[adDicts[index]["reportNo"]] = {}
    calc = Calc()

    for word_and_similarity in results:
        similarWord = word_and_similarity[0]
        cosineSimilarity = word_and_similarity[1]
        wordCount[similarWord] = 0
        if not is_noun(similarWord): continue
        for index in adDicts:
            report = adDicts[index]
            if is_few_words(report['advice_divide_mecab']): continue
            if not report['advice']: continue
            report_no = report["reportNo"]
            jsdDictionary[report_no] = calc.jsd(equation_lda_value, np.array(report['topic']))
            if similarWord not in report['advice_divide_mecab']: continue
            if is_not_match_report(report["companyType"], report["companyShokushu"]): continue
            wordDictionary[report_no].update({decode_word(similarWord): cosineSimilarity})
            if similarWord in report['tfidf']:
                similarity = report['tfidf'][similarWord] * cosineSimilarity
            else:
                similarity = cosineSimilarity
            rateCount.append([report_no, report["companyName"], similarity])
            reportNoType[report_no] = report["companyType"]
            reportNoShokushu[report_no] = report["companyShokushu"]
            lda[report_no] = report['topic']
            wordCount[similarWord] += 1

    [wordCount.pop(w[0]) for w in list(wordCount.items()) if w[1] == 0]


    # jsdは非類似度が高いほど値が大きくなるので、値が大きいほど類似度が高くなるように修正
    jsdDictionary = normalization(jsdDictionary)
    jsdDictionary = calc.value_reverse(jsdDictionary)

    # 同じ企業名で類似度を合計する
    fno1Comp = collections.Counter([comp[0] for comp in rateCount])

    rateCountNp = np.array(rateCount)
    reportNp = rateCountNp[:, [0]].reshape(-1,)
    nameSimilarityNp = rateCountNp[:, [1, 2]]

    compRecommendDic = {}

    for report_no in fno1Comp:
        typeRate = list_checked(reportNoType[report_no], company_type_name)
        shokushuRate = list_checked(reportNoShokushu[report_no], company_shokushu_name)
        similarSum = nameSimilarityNp[np.where(reportNp == str(report_no))]
        simSum = calcSimSum(similarSum)
        simLog = calcSimLog(simSum)
        compRecommendDic[report_no] = simLog + jsdDictionary[report_no] * (typeRate * shokushuRate)

    sortRecommendLevelReports = sorted(compRecommendDic.items(), key=lambda x: x[1], reverse=True)
    advice_json = {}
    for ranking, primaryComp in enumerate(sortRecommendLevelReports[:DISPLAY_REPORTS_NUM], start=1):
        report_no = primaryComp[0]
        advice_json[str(ranking)] = {
                'report_no': report_no,
                'recommend_level': str(round(primaryComp[1], DECIMAL_POINT)),
                'words': wordDictionary[report_no],
                'lda1': round(lda[report_no][0], DECIMAL_POINT),
                'lda2': round(lda[report_no][1], DECIMAL_POINT),
            }
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
equation = change_word(sys.argv[2])
company_type_name = sys.argv[3].split()
company_shokushu_name = sys.argv[4].split()
det_check = sys.argv[5]

if __name__=="__main__":
    similarReports = calc(equation)
    print(similarReports)


