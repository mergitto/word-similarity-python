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
    dict_cur = connect_pg()
    adviceDict = {}
    dict_cur.execute(SQL["QUERY"])
    for index, row in enumerate(dict_cur):
        report = dict(row)
        if report["advice"] != None:
            report["advice"] = clensing(report["advice"])
        else:
            report["advice"] = ''
        adviceDict[index] = {
                'reportNo': report["report_no"],
                "companyName": report["company_name"].replace("\u3000", " "),
                "companyType": report["type_name"],
                "companyShokushu": report["shokushu_name"],
                "advice": report["advice"],
                "course_code": report["course_name"],
                "advice_divide_mecab": '' if len(report["advice"]) == 0 else parser_mecab(report["advice"]),
                "advice_divide_mecab_space": '' if len(report["advice"]) == 0 else parser_space(report["advice"]),
                "report_created_date": report["created"],
                "info_date": report["indi_date"],
                "write_date": report["written_date"],
                "first_date": report["fir_oral_date"],
                "second_date": report["sec_oral_date"],
                "final_date": report["fin_oral_date"],
                "decision_date": report["decision_date"],
        }
    return adviceDict

def dictToPickle():
    CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
    advice = allAdvise()
    with open(CURRENTPATH+"/../pickle/advice.pickle", "wb") as f:
        pickle.dump(advice, f)

print("Get Advice")
dictToPickle()
print("Get Advice Finished!")

