import pickle
import pandas as pd
import numpy as np
import os
from const import *
from parse import parser_mecab
from preprocessing.tfidf import *
from replace import change_word, decode_word
# 日付に対応する
from datetime import datetime


CURRENTPATH = os.path.dirname(os.path.abspath(__file__))

def load_pickle():
    with open(CURRENTPATH+"/../pickle/advice_add_tfidf.pickle", 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data

def add_value(current_series, add_key_name):
    advice = load_pickle()
    for report in advice.values():
        report_no = report['reportNo']
        if current_series.report_no == report_no:
            current_series[add_key_name] = report[add_key_name]

def add_values(series):
    add_value(series, "advice_divide_mecab")
    add_value(series, "tfidf_top_average")
    add_value(series, "tfidf_sum")
    add_value(series, "topic")
    add_value(series, "bm25_sum")
    add_value(series, "bm25_average")
    add_value(series, "info_date")
    add_value(series, "write_date")
    add_value(series, "first_date")
    add_value(series, "second_date")
    add_value(series, "final_date")
    add_value(series, "decision_date")
    add_value(series, "report_created_date")
    return series

def is_not_nan(obj):
    return obj == obj

def perfect_check(target, keywords):
    is_in = False
    for keyword in keywords.split(','):
        if target == keyword:
            is_in = True
    return is_in

def is_match_keywords(search_word_wakati, keywords):
    if not is_not_nan(keywords): return False
    for search_word in search_word_wakati:
        if perfect_check(search_word, keywords):
            return True
    return False

def count_identification(document):
    # 一人称の単語の出現回数を返す
    from collections import Counter
    IDENTIFICATE_WORD = ["わたし", "私", "僕", "ぼく", "俺", "おれ", "自分", "じぶん"]
    document = Counter(document)
    ident_count = 0
    for word in document:
        for ident_word in IDENTIFICATE_WORD:
            if word == ident_word:
                ident_count += document[word]
    return ident_count

def is_exist_date(date):
    try:
        datetime.strptime(date, "%Y-%m-%d")
        return True
    except :
        return False

def to_date(date):
    return datetime.strptime(date, "%Y-%m-%d")

def count_selection(series):
    # 選考の回数をカウントする →　"write_date", "first_date", "second_date", "final_date"
    count_selection = 0
    for key in ["write_date", "first_date", "second_date", "final_date"]:
        if is_exist_date(series[key]):
            count_selection += 1
    return count_selection

def oral_first_final_diff_days(series):
    # 選考の最初の日付と内定獲得日の差分を取得するために"info_date"とか"decision_date"は使用する
    first_final_oral_days = 0
    if series.count_selection == 0: return first_final_oral_days
    first_oral_date = to_date("2100-04-01")
    final_oral_date = to_date("2008-04-01")
    for key in ["write_date", "first_date", "second_date", "final_date"]:
        if is_exist_date(series[key]):
            date = to_date(series[key])
            if final_oral_date < date:
                final_oral_date = date
            if first_oral_date > date:
                first_oral_date = date
        if is_exist_date(series.decision_date):
            final_oral_date = to_date(series.decision_date)
    first_final_oral_days = (final_oral_date - first_oral_date).days
    return first_final_oral_days

def top_similality_of_keywords(input_word, keywords):
    model_file_name = "~/Develop/Python/word-similarity-python/model/webdb/cpf/0_cpf.model"
    word2vecModel = load_word2vec_model(model_file_name)
    results = dict(get_similar_words(input_word, word2vecModel))
    similarity = {"top": 0, "sum": 0}
    if not is_not_nan(keywords): return similarity
    keywords = keywords.split(",")
    for keyword in keywords:
        keyword = change_word(keyword.lower())
        if keyword in results:
            if similarity["top"] < results[keyword]:
                similarity["top"] = results[keyword]
            similarity["sum"] += results[keyword]
    return similarity

def date_pluck(date):
    try:
        date = datetime.strptime(date, "%Y-%m-%d")
        if date.month < 10:
            month = "0"+str(date.month)
        else:
            month = str(date.month)
        date = str(date.year) + str(month)
    except:
        date = "111111"
    return int(date)

def diff_days_from_now(date):
    now = datetime.now()
    if is_exist_date(date):
        date = to_date(date)
    else:
        date = to_date("1111-11-01")
    diff_date = (now - date).days
    return diff_date if diff_date < 7000 else 7000

def add_column_df(dataframe):
    df = pd.DataFrame()
    for i in dataframe.iterrows():
        print("first: ", i[0]+1, "/", len(dataframe))
        series = i[1]
        series = add_values(series)
        series["report_created_datetime"] = date_pluck(series.report_created_date)
        series["today_created_diff_days"] = diff_days_from_now(series.report_created_date)
        series["count_selection"] = count_selection(series)
        series["first_final_diff_days"] = oral_first_final_diff_days(series)
        series["search_word_wakati"] = parser_mecab(str(series.search_word))
        series['word_length'] = len(series["advice_divide_mecab"])
        series["is_match_keywords"] = 1 if is_match_keywords(series.search_word_wakati, series.keywords) else 0
        similarity = top_similality_of_keywords(series.search_word_wakati, series.keywords)
        series["most_highest_similarity"] = similarity["top"]
        series["similarity_sum"] = similarity["sum"]
        series["identification_word_count"] = count_identification(series.advice_divide_mecab)
        df = df.append(series, ignore_index=True)
    return df

def score_norm(dataframe):
    st_no_keys = list(set(list(dataframe.st_no)))
    df = pd.DataFrame()
    for st_no_key in st_no_keys:
        st_no_df = dataframe[dataframe["st_no"] == st_no_key]
        score_df = st_no_df.score
        score_values = list(set(list(st_no_df.score)))
        if len(score_values) <= 10: continue
        for i in st_no_df.iterrows():
            series = i[1]
            score_min_max = ( series.score - score_df.min() ) / ( score_df.max() - score_df.min() )
            score_min_max = round(score_min_max, 2)
            series["score_min_max"] = score_min_max
            score_std = ( series.score - score_df.mean() ) / score_df.std()
            score_std = round(score_std, 2)
            series["score_std"] = score_std
            df = df.append(series, ignore_index=True)
    return df

def load_word2vec_model(model_file_name):
    from gensim.models import word2vec
    return word2vec.Word2Vec.load(model_file_name)

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
    for index, word in enumerate(inputWord): # 入力文字から類似語を出力
        word = change_word(word.lower())
        try:
            #if is_exist_input_word(word, model): results.append((word, INPUTWEIGHT))
            result = model.most_similar(positive = word, negative = [], topn = NEIGHBOR_WORDS)
            results = high_similar_words(result, results)
        except  Exception as e:
            pass
    return results


if True:
    questions = pd.read_csv('./questionnaire_all_evaluations_from_20181030.csv')
    df = add_column_df(questions)
    df = score_norm(df)
    df.to_csv('questionnaire_all_evaluations_preprocessed_from_20181030.csv', index=None)

