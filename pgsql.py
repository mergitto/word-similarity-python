# coding: UTF-8

# postgresqlと接続して、類似度の高い単語を含んだ報告書を取り出す
import psycopg2
import psycopg2.extras

import mojimoji
import re
from const import *

conn = psycopg2.connect(
    "host="+POSTGRES["PGHOST"]+" port="+POSTGRES["PORT"]+" dbname="+POSTGRES["DBNAME"]+" user="+POSTGRES["USER"]
)
conn.set_client_encoding('UTF8') # 文字コードの設定
dict_cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # 列名を指定してデータの取得を行う前準備

def allAdvise():
    dict_cur.execute(SQL["QUERY"])
    adviceDict = {}
    for index, row in enumerate(dict_cur):
        if row[3] != None:
            row[3] = re.sub("\<.+?\>", "", row[3])
            row[3] = row[3].lower()
            row[3] = re.sub("\[.+?\]", "", row[3])
            row[3] = mojimoji.han_to_zen(mojimoji.zen_to_han(row[3], kana=False, ascii=False), digit=False) # 数字だけ半角で、カナとローマ字は全角
            # 同意義語の表記統一
            row[3] = re.sub("ｂｅｓｔ", "ベスト",row[3])
            row[3] = re.sub("ｓｕｃｃｅｓｓｓｑｉ", "サクセスｓｑｉ",row[3])
            row[3] = re.sub("ｅｌｓｅ", "ｅｌｓ",row[3])
            row[3] = re.sub("ｏｐｅｎｅｓ", "エントリーシート",row[3])
            row[3] = re.sub("ｏｐｅｎ　ｅｓ", "エントリーシート",row[3])
            row[3] = re.sub("ｏｅｓ", "エントリーシート",row[3])
            row[3] = re.sub("ｅｓ", "エントリーシート",row[3])
            row[3] = re.sub("ｓｅ", "システムエンジニア",row[3])
            row[3] = re.sub("ｇｄ", "グループディスカッション",row[3])
            row[3] = re.sub("ｈｐ", "ホームページ",row[3])
            row[3] = re.sub("ピーアール", "ｐｒ",row[3])
            row[3] = re.sub("ｐｇ", "プログラマー",row[3])
            row[3] = re.sub("ｇｃ", "ゲームクリエイター",row[3])
            row[3] = re.sub("ウェブ", "ｗｅｂ",row[3])
            row[3] = re.sub("コミュニケーション力", "コミュニケーション能力",row[3])
            row[3] = re.sub("コニニケーション", "コミュニケーション",row[3])
            row[3] = re.sub("コミュニティーション", "コミュニケーション",row[3])
            row[3] = re.sub("かんばる", "頑張る",row[3])
            row[3] = re.sub("がんばる", "頑張る",row[3])
            row[3] = re.sub("かんばって", "頑張って",row[3])
            row[3] = re.sub("ｇｐａ", "ｇｐａ ",row[3]) # gpa3.? の場合に gp a3 で分かち書きされるためにgpaの後に空白追加
            row[3] = re.sub("ｉｔ", "ｉｃｔ",row[3]) # ictの方が現代の言葉なので表記揺れ回避
            row[3] = mojimoji.zen_to_han(row[3], kana=False, digit=False)
            # 単語の英字1〜2文字以下の場合は削除する 例：I am student. -> I, am は削除する
            row[3] = re.sub("[ ][a-z]{1,2}[ ]", "",row[3])
            # ( )で囲まれた部分を削除する 例：<br />
            row[3] = re.sub("\(.+?\)", "",row[3])
        adviceDict[index] = {'reportNo': row[4], "companyName": row[1], "companyType": row[2],"advice": row[3]}
    return adviceDict
