import pandas as pd
import pickle
from preprocessing.random_forest import SupervisedLearning
import os

def load_pickle(load_file_name=None):
    with open(load_file_name, 'rb') as f:
        return pickle.load(f)

def dump_pickle(dump_data, dump_file_name=None):
    with open(dump_file_name, 'wb') as f:
        pickle.dump(dump_data, f)

drop_list = [
        "report_created_date", "bm25_sum", "bm25", "advice",
        "type_id", "shokushu_id", "score_std", "recommend_level",
        "info_date", "write_date", "first_date", "second_date", "final_date", "decision_date",
        "advice_divide_mecab", "bm25_average", "course_code", "created",
        "keywords", "modified", "score", "search_word", "search_word_wakati", "st_no",
        "most_highest_similarity", "is_good", "recommend_formula", "evaluation_id", "report_no",
        "similarity_sum", "is_match_keywords", "recommend_rank",
        #"tfidf_top_average", "report_created_datetime", "ne_word_count",
        #'count_selection', 'today_created_diff_days, 'first_final_diff_days',
        #'identification_word_count', 'tfidf_sum', 'word_length', 'word_count',
    ]

CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
evaluations_data = load_pickle(load_file_name=CURRENTPATH+"/../pickle/evaluations_add_values.pickle")

print("Create RandomForestModel")
tree = SupervisedLearning(evaluations_data)
tree.add_dummy_score(high_report_rate=0.4)
tree.drop_na(drop_na_list=["score", "score_std", "recommend_rank"])
tree.drop_columns(drop_list)
tree.set_X_and_y(objective_key="score_dummy")

max_depth = 4
tree.random_forest(max_depth=max_depth)
tree.save_model(save_model_name=CURRENTPATH+"/../pickle/random_forest.model")
print("Create RandomForestModel finished!")

print("Add Predicted(random_forest) Start")
clf = tree.load_model(load_model_name=CURRENTPATH+"/../pickle/random_forest.model")
reports = load_pickle(load_file_name=CURRENTPATH+"/../pickle/advice_add_values_for_machine_learning.pickle")
reports = tree.add_predicted(clf=clf, pickle_data=reports,
        is_high_predicted_name="is_high_predicted", predicted_high_rate_name="predicted_high_rate")
dump_pickle(reports, dump_file_name=CURRENTPATH+"/../pickle/advice_add_predicted.pickle")
print("Add Predicted Finished!")

print("Get Importances")
importance_dict = tree.clf_importance(tree.X, tree.clf)
dump_pickle(importance_dict, dump_file_name=CURRENTPATH+"/../pickle/importance_dict.pickle")
print("Get Importances Finished!")

print("Add Predicted(neural) Start")
tree.neural()
tree.save_model(save_model_name=CURRENTPATH+"/../pickle/neural_mlp.model")
clf = tree.load_model(load_model_name=CURRENTPATH+"/../pickle/neural_mlp.model")
reports = load_pickle(load_file_name=CURRENTPATH+"/../pickle/advice_add_predicted.pickle")
reports = tree.add_predicted(clf=clf, pickle_data=reports,
        is_high_predicted_name="is_high_predicted_neural", predicted_high_rate_name="predicted_neural_high_rate")
dump_pickle(reports, dump_file_name=CURRENTPATH+"/../pickle/advice_add_predicted_neural.pickle")
print("Add Predicted Finished!")

print("Add Predicted(svm) Start")
tree.svm()
tree.save_model(save_model_name=CURRENTPATH+"/../pickle/svm_clf.model")
clf = tree.load_model(load_model_name=CURRENTPATH+"/../pickle/svm_clf.model")
reports = load_pickle(load_file_name=CURRENTPATH+"/../pickle/advice_add_predicted_neural.pickle")
reports = tree.add_predicted(clf=clf, pickle_data=reports,
        is_high_predicted_name="is_high_predicted_svm", predicted_high_rate_name="predicted_svm_high_rate")
dump_pickle(reports, dump_file_name=CURRENTPATH+"/../pickle/advice_add_predicted_svm.pickle")
print("Add Predicted Finished!")

print("Add Importance Rate")
importances = load_pickle(load_file_name=CURRENTPATH+"/../pickle/importance_dict.pickle")
reports = load_pickle(load_file_name=CURRENTPATH+"/../pickle/advice_add_predicted_svm.pickle")
reports = tree.add_importances_rate(importances=importances, pickle_data=reports)
dump_pickle(reports, dump_file_name=CURRENTPATH+"/../pickle/advice_add_importances_rate.pickle")
print("Add Importance Rate Finished!")

