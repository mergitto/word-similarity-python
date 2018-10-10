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
from const import *

#定数の宣言
similaryty = 0.50 # 類似度を設定する
INPUTWEIGHT = 1.0 # 入力文字の重み（仮想的な類似度）
############


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

def get_similar_words(inputWord, model):
    results = []
    inputVectorSum = 0
    for index, word in enumerate(inputWord): # 入力文字から類似語を出力
        try:
            if is_exist_input_word(word, model): results.append((word, INPUTWEIGHT))
            result = model.most_similar(positive = word, negative = [], topn = NEIGHBOR_WORDS)
            results = high_similar_words(result, results)
        except  Exception as e:
            pass
        inputVectorSum += model[word]
    return results, inputVectorSum

def load_reports():
    with open(PATH["REPORTS_PICKELE"], 'rb') as f: # トピック分類の情報を付加したデータをpickleでロード
        adDicts = pickle.load(f)
    return adDicts

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

def neighbor_word(posi, nega=[], n=300, inputText = None):
    posi = sorted(list(set(posi)), key=posi.index)
    results, inputVectorSum = get_similar_words(posi, model)
    inputVectorLength = np.linalg.norm(inputVectorSum)
    rateCount = []
    wordDictionary = {} # 報告書ごとの類似単語辞書
    jsdDictionary = {} # 報告書ごとに入力とldaのtopic値を活用してjsd値を計算する
    cosSimilar = {}
    highlowDictionary = {} # 検証用の高評価・低評価判定用の辞書
    equation_lda_value = np.array(lda_value(equation, [posi])['topic']) # 入力値にLDAによるtopic値を付与する

    adDicts = load_reports()
    for index in adDicts:
        wordDictionary[adDicts[index]["reportNo"]] = {}
    calc = Calc()

    for word_and_similarity in results:
        similarWord = word_and_similarity[0]
        cosineSimilarity = word_and_similarity[1]
        if not is_noun(similarWord): continue
        for index in adDicts:
            report = adDicts[index]
            if is_few_words(report['advice_divide_mecab']): continue
            if not report['advice']: continue
            report_no = report["reportNo"]
            jsdDictionary[report_no] = calc.jsd(equation_lda_value, np.array(report['topic']))
            if similarWord not in report['advice_divide_mecab']: continue
            wordDictionary[report_no].update({decode_word(similarWord): cosineSimilarity})
            if similarWord in report['tfidf']:
                similarity = report['tfidf'][similarWord] * cosineSimilarity
            else:
                similarity = cosineSimilarity
            highlowDictionary[report_no] = report['evaluation']
            cosSimilar[report_no] = np.dot(report['vectorSum'], inputVectorSum) / (report['vectorLength'] * inputVectorLength)
            rateCount.append([report_no, report["companyName"], similarity])

    # 内積の計算でコサイン類似度がマイナスになることがあったので、正規化した
    cosSimilar = normalization(cosSimilar)
    jsdDictionary = normalization(jsdDictionary)
    jsdDictionary = calc.value_reverse(jsdDictionary)

    reportDict = {} # 類似語を含むアドバイスの類似度をreport_no毎に足し算する
    # 同じ企業名で類似度を合計する
    fno1Comp = collections.Counter([comp[0] for comp in rateCount])

    rateCountNp = np.array(rateCount)
    reportNp = rateCountNp[:, [0]].reshape(-1,)
    nameSimilarityNp = rateCountNp[:, [1, 2]]

    compRecommendDic = {}

    for comp_no in fno1Comp:
        similarSum = nameSimilarityNp[np.where(reportNp == str(comp_no))]
        simSum = calcSimSum(similarSum)
        simLog = calcSimLog(simSum)
        compRecommendDic[comp_no] = simLog + cosSimilar[comp_no] * jsdDictionary[comp_no]


    advice_json = {}
    for index, primaryComp in enumerate(sorted(compRecommendDic.items(), key=lambda x: x[1], reverse=True)[:20]):
        ranking = index + 1
        advice_json[str(ranking)] = {
                'report_no': primaryComp[0],
                'recommend_level': str(round(primaryComp[1], 3)),
                'evaluation': highlowDictionary[primaryComp[0]],
            }
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
if __name__=="__main__":
    equation = change_word(sys.argv[2])
    similarReports = calc(equation)
    pprint(similarReports, width=80)


