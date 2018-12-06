# -*- coding: utf-8 -*-

# postgresqlと接続して、類似度の高い単語を含んだ報告書を取り出す
import psycopg2
import psycopg2.extras
import mojimoji
import re
from const import *
import pickle
from parse import *
import os

def connect_pg():
    conn = psycopg2.connect(
        "host="+POSTGRES["PGHOST"]+" port="+POSTGRES["PORT"]+" dbname="+POSTGRES["DBNAME"]+" user="+POSTGRES["USER"]
    )
    conn.set_client_encoding('UTF8') # 文字コードの設定
    dict_cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # 列名を指定してデータの取得を行う前準備
    return dict_cur

def get_evaluations():
    dict_cur = connect_pg()
    dict_cur.execute(SQL_EVALUATIONS["QUERY"])
    evaluationDict = {}
    for index, row in enumerate(dict_cur):
        evaluation = dict(row)
        evaluationDict[index] = evaluation
    return evaluationDict

def dictToPickle():
    CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
    evaluations = get_evaluations()
    with open(CURRENTPATH+"/../pickle/evaluations.pickle", "wb") as f:
        pickle.dump(evaluations, f)

print("Get Evaluations")
dictToPickle()
print("Get Evaluations Finished!")

