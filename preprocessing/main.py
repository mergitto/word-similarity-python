import pandas as pd
import pickle
from preprocessing.random_forest import Tree
import os

def load_pickle(load_file_name=None):
    with open(load_file_name, 'rb') as f:
        return pickle.load(f)

def dump_pickle(dump_data, dump_file_name=None):
    with open(dump_file_name, 'wb') as f:
        pickle.dump(dump_data, f)

drop_list = [
        "report_created_date", "bm25_sum", "bm25", "advice",
        "type_id", "shokushu_id", "score_std",
        "recommend_level", "tfidf_top_average",
        "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
        "advice_divide_mecab", "bm25_average", "course_code", "created",
        "keywords", "modified", "score", "score_min_max", "search_word",
        "search_word_wakati", "st_no",
        "most_highest_similarity", "is_good", "recommend_formula", "evaluation_id", "report_no",
        "similarity_sum", "report_created_datetime", "is_match_keywords", "recommend_rank",
        #'count_selection', 'today_created_diff_days, 'first_final_diff_days',
        #'identification_word_count', 'tfidf_sum', 'word_length', 'word_count',
    ] # 不必要なカラム

CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
evaluations_data = load_pickle(load_file_name=CURRENTPATH+"/../pickle/evaluations_add_values.pickle")

print("Create RandomForestModel")
tree = Tree(evaluations_data)
tree.add_dummy_score(high_report_rate=0.4)
tree.drop_na(drop_na_list=["score", "recommend_rank"])
tree.drop_columns(drop_list)
tree.set_X_and_y(objective_key="score_dummy")

max_depth = 4
tree.random_forest(max_depth=4)
tree.save_model(save_model_name=CURRENTPATH+"/../pickle/random_forest.model")
print("Create RandomForestModel finished!")

print("Add Predicted Start")
clf = tree.load_model(load_model_name=CURRENTPATH+"/../pickle/random_forest.model")
all_reports = load_pickle(load_file_name=CURRENTPATH+"/../pickle/advice_add_values_for_machine_learning.pickle")
all_reports_with_predicted = tree.add_predicted(clf=clf, pickle_data=all_reports)
dump_pickle(all_reports_with_predicted, dump_file_name=CURRENTPATH+"/../pickle/advice_add_predicted.pickle")
print("Add Predicted Finished!")


