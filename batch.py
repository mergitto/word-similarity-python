# データ取得＆推薦用のデータ前処理とデータ追加のバッチ処理
from preprocessing import pgsql
from preprocessing import addVector
from preprocessing import tfidf

# 機械学習による前学習用のデータ追加処理
from preprocessing import pgsql_evaluations
from preprocessing import add_values

# random_forestで学習&モデルの保存
from preprocessing import main

