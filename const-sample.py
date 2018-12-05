POSTGRES = {
    "PGHOST": "",
    "PORT": "",
    "DBNAME": "",
    "USER": "",
}

sql = ""
SQL = {
    "QUERY": sql,
}
SQL_EVALUATIONS = {
    "QUERY": "",
}
PATH = {
    # 分かち書きしたcsvファイルを指定する(mecab, jumanpp)
    "WAKACHI": "",
    # 報告書の情報を詰め込んだやつ(addTopic.pyのトピック数と合わせなければならない)
    "REPORTS_PICKELE": "",
}

SIMILARYTY_LIMIT_RATE = 0.50 # 類似度を設定する
INPUTWEIGHT = 1.0 # 入力文字の重み（仮想的な類似度）
DISPLAY_REPORTS_NUM = 100 # 表示する報告書数
DECIMAL_POINT = 3 # 小数点第何位までにする
LOWEST_WORD_LENGTH = 10 # 推薦の計算をする報告書の最低単語数
NEIGHBOR_WORDS = 300 # 類似単語の上位n単語分の使用する
WORD2VECMODELFILE = "your model file path" # word2vecのモデルファイルの指定

