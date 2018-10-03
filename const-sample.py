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
PATH = {
    # 分かち書きしたcsvファイルを指定する(mecab, jumanpp)
    "WAKACHI": "",
    # 報告書の情報を詰め込んだやつ(addTopic.pyのトピック数と合わせなければならない)
    "REPORTS_PICKELE": "",
}

# 類似度を設定する
SIMILARYTY_LIMIT_RATE = 0.50
# 入力文字の重み（仮想的な類似度）
INPUTWEIGHT = 1.0
