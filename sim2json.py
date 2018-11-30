# -*- coding: utf-8 -*-
#!/usr/local/pyenv/shims/python

from gensim.models import word2vec
import sys
import collections
from collections import defaultdict
import numpy as np
import pickle
import math
from parse import parser_mecab
from parse import is_noun
from replace import change_word
from replace import decode_word
from calc import Calc
from const import *
from json_extend import json_dump

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
    simSum = 0.1
    if len(similarSumary) > 0:
        for similarity in similarSumary:
            simSum += similarity
    return simSum

def calcSimLog(simSum):
    simLog = 0.0001 if math.log(simSum, 2) <= 0 else math.log(simSum, 2)
    return simLog * 1.2

def is_few_words(parse_text):
    if len(parse_text) < LOWEST_WORD_LENGTH:
        return True
    return False

def clean_sort_dictionary(dictionary):
    [dictionary.pop(w[0]) for w in list(dictionary.items()) if w[1] == 0]
    return sorted(dictionary.items(), key=lambda x:x[1], reverse=True)

def advice_to_json(recommend_dict, reports_values, word_count):
    sortRecommendLevelReports = sorted(recommend_dict.items(), key=lambda x: x[1], reverse=True)
    advice_json = {}
    for ranking, primaryComp in enumerate(sortRecommendLevelReports[:DISPLAY_REPORTS_NUM], start=1):
        report_no = primaryComp[0]
        advice_json[str(ranking)] = {
                'report_no': report_no,
                'recommend_level': str(round(primaryComp[1], DECIMAL_POINT)),
                'words': reports_values[report_no]["word_and_similarity"],
            }
    advice_json['word_count'] = word_count
    advice_json['company_type'] = company_type_name
    advice_json['company_shokushu'] = company_shokushu_name
    return advice_json

def recommend_rate(reports_values):
    compRecommendDic = {}
    for report_no in reports_values:
        typeRate = list_checked(reports_values[report_no]["type"], company_type_name)
        shokushuRate = list_checked(reports_values[report_no]["shokushu"], company_shokushu_name)
        simSum = calcSimSum(reports_values[report_no]["similarities"])
        simLog = calcSimLog(simSum)
        recommend_rate = simSum * (typeRate * shokushuRate)
        compRecommendDic[report_no] = recommend_rate
    return compRecommendDic

def initialize_report_dict(advice_dictionary):
    reports_values = {}
    for index in advice_dictionary:
        report_no = advice_dictionary[index]["reportNo"]
        reports_values[report_no] = {}
        reports_values[report_no]["similarities"] = []
        reports_values[report_no]["word_and_similarity"] = {}
        reports_values[report_no]["type"] = None
        reports_values[report_no]["shokushu"] = None
    return reports_values

def neighbor_word(posi, nega=[], n=NEIGHBOR_WORDS, inputText = None):
    posi = sorted(list(set(posi)), key=posi.index)
    results = get_similar_words(posi)

    wordCount = {} # 類似単語の出現回数

    adDicts = load_reports()
    reports_values = initialize_report_dict(adDicts)

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
            if is_not_match_report(report["companyType"], report["companyShokushu"]): continue
            if similarWord in report['tfidf']:
                similarity = report['tfidf'][similarWord] * cosineSimilarity
            else:
                similarity = cosineSimilarity
            if similarWord not in report['advice_divide_mecab']:
                similarity = 0.0001
            reports_values[report_no]["similarities"].append(similarity)
            reports_values[report_no]["type"] = report["companyType"]
            reports_values[report_no]["shokushu"] = report["companyShokushu"]

            if similarWord not in report['advice_divide_mecab']: continue
            wordCount[similarWord] += 1
            reports_values[report_no]["word_and_similarity"].update({decode_word(similarWord): cosineSimilarity})

    wordCount = clean_sort_dictionary(wordCount)

    recommendRateDict = recommend_rate(reports_values)
    advice_json = advice_to_json(recommendRateDict, reports_values, wordCount)

    return json_dump(advice_json)

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


