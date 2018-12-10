import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from const import *
from parse import parser_mecab
from preprocessing.tfidf import *
from replace import change_word, decode_word

class AddValues():
    def __init__(self, CURRENTPATH=""):
        self.CURRENTPATH = CURRENTPATH
        self.reports = self.load_pickle(load_path=CURRENTPATH+"/../pickle/advice_add_tfidf.pickle")
        self.evaluations = self.load_pickle(load_path=CURRENTPATH+"/../pickle/evaluations.pickle")
        self.word2vecModel = self.load_word2vec_model(model_file_name=WORD2VECMODELFILE)

    def load_pickle(self, load_path=""):
        with open(load_path, "rb") as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def dump_pickle(self, dump_data=None, dump_path=""):
        with open(dump_path, "wb") as f:
            pickle.dump(dump_data, f)

    def load_word2vec_model(self, model_file_name):
        from gensim.models import word2vec
        return word2vec.Word2Vec.load(model_file_name)

    def to_date(self, date):
        return datetime.strptime(date, "%Y-%m-%d")

    def date_pluck(self, date):
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

    def is_exist_date(self, date):
        try:
            datetime.strptime(date, "%Y-%m-%d")
            return True
        except :
            return False

    def diff_days_from_now(self, date):
        now = datetime.now()
        if self.is_exist_date(date):
            date = self.to_date(date)
        else:
            date = self.to_date("1111-11-01")
        diff_date = (now - date).days
        return diff_date if diff_date < 7000 else 7000

    def count_selection(self, dictionary):
        # 選考の回数をカウントする →　"write_date", "first_date", "second_date", "final_date"
        count_selection = 0
        for key in ["write_date", "first_date", "second_date", "final_date"]:
            if self.is_exist_date(dictionary[key]):
                count_selection += 1
        return count_selection

    def oral_first_final_diff_days(self, dictionary):
        # 選考の最初の日付と内定獲得日の差分を取得するために"info_date"とか"decision_date"は使用する
        first_final_oral_days = 0
        if dictionary["count_selection"] == 0: return first_final_oral_days
        first_oral_date = self.to_date("2100-04-01")
        final_oral_date = self.to_date("2008-04-01")
        for key in ["write_date", "first_date", "second_date", "final_date"]:
            if self.is_exist_date(dictionary[key]):
                date = self.to_date(dictionary[key])
                if final_oral_date < date:
                    final_oral_date = date
                if first_oral_date > date:
                    first_oral_date = date
            if self.is_exist_date(dictionary["decision_date"]):
                final_oral_date = self.to_date(dictionary["decision_date"])
        first_final_oral_days = (final_oral_date - first_oral_date).days
        return first_final_oral_days

    def count_identification(self, document):
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

    def score_norm(self):
        st_no_keys = list(set([self.evaluations[report_no]["st_no"] for report_no in self.evaluations]))
        df = pd.DataFrame()
        dataframe = pd.DataFrame.from_dict(self.evaluations)
        dataframe = dataframe.T
        dataframe = dataframe.dropna(subset=["score"])
        for st_no_key in st_no_keys:
            st_no_df = dataframe[dataframe["st_no"] == st_no_key]

            score_df = st_no_df.score
            score_values = list(set(list(st_no_df.score)))
            for i in st_no_df.iterrows():
                series = i[1]
                score_min_max = ( series.score - score_df.min() ) / ( score_df.max() - score_df.min() )
                score_min_max = round(score_min_max, 2)
                series["score_min_max"] = score_min_max
                score_std = ( series.score - score_df.mean() ) / score_df.std()
                score_std = round(score_std, 2)
                series["score_std"] = score_std
                df = df.append(series, ignore_index=True)

        report_numbers = list(df.report_no)
        copy_eval = self.evaluations.copy()
        for index in copy_eval:
            c_eval = copy_eval[index]
            if c_eval["report_no"] not in report_numbers:
                del self.evaluations[index]

        for index in self.evaluations:
            c_report = self.evaluations[index]
            match_df = df[df["evaluation_id"] == c_report["evaluation_id"]]
            match_df =  match_df.iloc[-1]
            c_report["score_std"] = match_df.score_std
            c_report["score_min_max"] = match_df.score_min_max

    def perfect_check(self, target, keywords):
        is_in = False
        for keyword in keywords.split(','):
            if target == keyword:
                is_in = True
        return is_in

    def is_match_keywords(self, search_word_wakati, keywords):
        if keywords is None: return False
        for search_word in search_word_wakati:
            if self.perfect_check(search_word, keywords):
                return True
        return False

    def add_columns(self):
        self.add_values()
        for index in self.evaluations:
            current_data = self.evaluations[index]
            current_data["report_created_datetime"] = self.date_pluck(current_data["report_created_date"])
            current_data["today_created_diff_days"] = self.diff_days_from_now(current_data["report_created_date"])
            current_data["count_selection"] = self.count_selection(current_data)
            current_data["first_final_diff_days"] = self.oral_first_final_diff_days(current_data)
            current_data["word_length"] = len(current_data["advice"])
            current_data["word_count"] = len(current_data["advice_divide_mecab"])
            current_data["identification_word_count"] = self.count_identification(current_data["advice_divide_mecab"])
            current_data["search_word_wakati"] = parser_mecab(str(current_data["search_word"]))
            current_data["is_match_keywords"] = 1 if self.is_match_keywords(current_data["search_word_wakati"], current_data["keywords"]) else 0
            similarity = self.top_similality_of_keywords(current_data["search_word_wakati"], current_data["keywords"])
            current_data["most_highest_similarity"] = similarity["top"]
            current_data["similarity_sum"] = similarity["sum"]

        for index in self.reports:
            current_report = self.reports[index]
            current_report["report_created_datetime"] = self.date_pluck(current_report["report_created_date"])
            current_report["count_selection"] = self.count_selection(current_report)
            current_report["today_created_diff_days"] = self.diff_days_from_now(current_report["report_created_date"])
            current_report["first_final_diff_days"] = self.oral_first_final_diff_days(current_report)
            current_report["identification_word_count"] = self.count_identification(current_report["advice_divide_mecab"])
            current_report["word_length"] = len(current_report["advice"])
            current_report["word_count"] = len(current_report["advice_divide_mecab"])

    def add_values(self):
        for i in self.evaluations:
            c_eval = self.evaluations[i]
            report_no_c_eval = c_eval["report_no"]
            add_key_list = [
                    "advice", "advice_divide_mecab", "tfidf_top_average", "tfidf_sum",
                    "bm25", "bm25_sum", "bm25_average", "report_created_date",
                    "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
                ]
            for index in self.reports:
                c_report = self.reports[index]
                if c_report["reportNo"] == report_no_c_eval:
                    for key in add_key_list:
                        c_eval[key] = c_report[key]

    def top_similality_of_keywords(self, input_word, keywords):
        results = dict(self.get_similar_words(input_word, self.word2vecModel))
        similarity = {"top": 0, "sum": 0}
        if keywords is None: return similarity
        keywords = keywords.split(",")
        for keyword in keywords:
            keyword = change_word(keyword.lower())
            if keyword in results:
                if similarity["top"] < results[keyword]:
                    similarity["top"] = results[keyword]
                similarity["sum"] += results[keyword]
        return similarity

    def high_similar_words(self, result, results):
        for r in result:
            if r[1] > SIMILARYTY_LIMIT_RATE:
                results.append(r)
            else:
                continue;
        return results

    def get_similar_words(self, inputWord, model):
        results = []
        for index, word in enumerate(inputWord): # 入力文字から類似語を出力
            word = change_word(word.lower())
            try:
                result = model.most_similar(positive = word, negative = [], topn = NEIGHBOR_WORDS)
                results = self.high_similar_words(result, results)
            except  Exception as e:
                pass
        return results

print("AddValues")
CURRENTPATH = os.path.dirname(os.path.abspath(__file__))

add_values = AddValues(CURRENTPATH=CURRENTPATH)
add_values.add_columns()
add_values.score_norm()
add_values.dump_pickle(dump_data=add_values.evaluations ,dump_path=CURRENTPATH+"/../pickle/evaluations_add_values.pickle")
add_values.dump_pickle(dump_data=add_values.reports,dump_path=CURRENTPATH+"/../pickle/advice_add_values_for_machine_learning.pickle")
print("AddValues finished!")

