# -*- coding: utf-8 -*-

# postgresqlと接続して、類似度の高い単語を含んだ報告書を取り出す
import psycopg2
import psycopg2.extras
import mojimoji
import re
from const import *
import pickle
from parse import *

conn = psycopg2.connect(
    "host="+POSTGRES["PGHOST"]+" port="+POSTGRES["PORT"]+" dbname="+POSTGRES["DBNAME"]+" user="+POSTGRES["USER"]
)
conn.set_client_encoding('UTF8') # 文字コードの設定
dict_cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # 列名を指定してデータの取得を行う前準備

def clensing(text):
    text = re.sub("\<.+?\>", "", text)
    text = text.lower()
    text = re.sub("\[.+?\]", "", text)
    text = mojimoji.han_to_zen(mojimoji.zen_to_han(text, kana=False, ascii=False), digit=False) # 数字だけ半角で、カナとローマ字は全角
    # 同意義語の表記統一
    text = re.sub("ｂｅｓｔ", "ベスト",text)
    text = re.sub("ｓｕｃｃｅｓｓｓｑｉ", "サクセスｓｑｉ",text)
    text = re.sub("ｅｌｓｅ", "ｅｌｓ",text)
    text = re.sub("ｏｐｅｎｅｓ", "エントリーシート",text)
    text = re.sub("ｏｐｅｎ　ｅｓ", "エントリーシート",text)
    text = re.sub("ｏｅｓ", "エントリーシート",text)
    text = re.sub("ｅｓ", "エントリーシート",text)
    text = re.sub("ｓｅ", "システムエンジニア",text)
    text = re.sub("ｇｄ", "グループディスカッション",text)
    text = re.sub("ｈｐ", "ホームページ",text)
    text = re.sub("ピーアール", "ｐｒ",text)
    text = re.sub("ｐｇ", "プログラマー",text)
    text = re.sub("ｇｃ", "ゲームクリエイター",text)
    text = re.sub("ウェブ", "ｗｅｂ",text)
    text = re.sub("コミュニケーション力", "コミュニケーション能力",text)
    text = re.sub("コニニケーション", "コミュニケーション",text)
    text = re.sub("コミュニティーション", "コミュニケーション",text)
    text = re.sub("かんばる", "頑張る",text)
    text = re.sub("がんばる", "頑張る",text)
    text = re.sub("かんばって", "頑張って",text)
    text = re.sub("ｇｐａ", "ｇｐａ ",text) # gpa3.? の場合に gp a3 で分かち書きされるためにgpaの後に空白追加
    text = re.sub("ｉｔ", "ｉｃｔ",text) # ictの方が現代の言葉なので表記揺れ回避
    text = mojimoji.zen_to_han(text, kana=False, digit=False)
    # 単語の英字1〜2文字以下の場合は削除する 例：I am student. -> I, am は削除する
    text = re.sub("[ ][a-z]{1,2}[ ]", "",text)
    # ( )で囲まれた部分を削除する 例：<br />
    text = re.sub("\(.+?\)", "",text)
    return text

def allAdvise():
    dict_cur.execute(SQL["QUERY"])
    adviceDict = {}
    select_count = len(dict_cur.fetchall())
    dict_cur.execute(SQL["QUERY"])
    for index, row in enumerate(dict_cur):
        print(round(index / select_count * 100, 3), "%")
        if row[3] != None:
            row[3] = clensing(row[3])
        else:
            row[3] = ''
        adviceDict[index] = {
                'reportNo': row[4],
                "companyName": row[1].replace("\u3000", " "),
                "companyType": row[2],
                "companyShokushu": row[5],
                "advice": row[3],
                "companyShokushu": row[5],
                "course_code": row[0],
                "advice_divide_mecab": '' if len(row[3]) == 0 else parser_mecab(row[3]),
                "advice_divide_mecab_space": '' if len(row[3]) == 0 else parser_space(row[3]),
        }
    return adviceDict

def dictToPickle():
    advice = allAdvise()
    with open("advice.pickle", "wb") as f:
        pickle.dump(advice, f)

if __name__ == '__main__':
    dictToPickle()
