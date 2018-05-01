# -*- coding: utf-8 -*-

import gensim
from gensim.models import word2vec
import sys
import collections
import numpy as np
import pickle
import compTypeList
import math
from parse import parser_mecab
import time

#定数の宣言
similaryty = 0.60 # 類似度を設定する
INPUTWEIGHT = 1.0 # 入力文字の重み（仮想的な類似度）
PRIORITYRATE = 5 # 重要単語を選択した時に付加する類似語の類似度の倍率
LOWPRIORITYRATE = 0.5 # 非重要単語を選択した時に付加する類似語の類似度の倍率
INPUTWORDPATH = './output/inputWord.txt' # 入力単語を保存するファイル
RECCOMPPATH = './output/enterprise.csv' # 推薦企業の上位を保存するファイル
EVALUATIONPATH = './output/evaluation.csv' # 推薦企業の評価を保存するファイル
INPUTTYPEPATH = './output/inputType.csv' # 推薦企業の評価を保存するファイル
WRITE = False # 入力内容を書き込むか否か Trueなら書き込み、Falseなら書き込まない
WEIGHTING = True # 入力文字のから重要単語を選択する場合はTrue,しない場合はFalse
TYPE = True
############

model   = word2vec.Word2Vec.load(sys.argv[1])

# bizreachのモデルを利用する場合は以下のコメントアウトを削除する
#from gensim.models import KeyedVectors
#MODEL_FILENAME = "model/bizreach.model"
#model = KeyedVectors.load_word2vec_format(MODEL_FILENAME, binary=True)
# LDAによるトピック分類を利用した推薦のためのモデル読み込み

def neighbor_word(posi, nega=[], n=300, inputText = None):
    tmpWordCheck = ''
    count = 0

    results = []
    inputVectorSum = 0 # 入力文字のベクトルの和を格納
    inputVectorLength = 0 # 入力文字のベクトル長を格納
    resultWord = [] # 入力文字の中でword2vecによって学習されている単語を格納する
    posi = sorted(list(set(posi)), key=posi.index)
    for inputWord in posi:
        try:
            result = model.most_similar(positive = inputWord, negative = nega, topn = n)
            resultWord.append(inputWord)
        except  Exception as e:
            continue
        results.append((inputWord, INPUTWEIGHT))
    posi = resultWord
    print('入力文字',posi)
    if WEIGHTING == True and ALGORITHMTYPE == 0:
        weightingFlag = compTypeList.weightingSimilar(posi)
    for index, po in enumerate(posi): # 入力文字から類似語を出力
        try:
            result = model.most_similar(positive = po, negative = nega, topn = n)
            tmpWordCheck += '1,' + po + ','
            for r in result:
                if r[1] > similaryty:
                    if WEIGHTING == True and not ALGORITHMTYPE == 0:
                        results.append(r)
                    else:
                        if index == int(weightingFlag): # 入力の中で重要であると利用者が判断した単語の類似語の類似度を少し増やす
                            results.append((r[0], r[1] * PRIORITYRATE))
                        else:
                            results.append((r[0], r[1] * LOWPRIORITYRATE))
                else:
                    break;
            # 入力のベクトルの和
            inputVectorSum += model[po]
        except  Exception as e:
            print('「' + po + '」という単語は見つからなかったです')
            tmpWordCheck += '0,' + po + ','
        count += 1
    inputVectorLength = np.linalg.norm(inputVectorSum)

    words = {'positive': posi, 'negative': nega}
    # adDictsのpickleをロードする
    with open('./tfidf/advice_10_tfidf.pickle', 'rb') as f: # トピック分類の情報を付加したデータをpickleでロード
        adDicts = pickle.load(f)
    rateCount = []
    topicDic = {} # 入力と文書ごとのトピック積和を格納
    cosSimilar = {} # 入力と文書ごとのコサイン類似度を格納
    reportNoType = {} # 報告書Noと業種の辞書
    for kensaku in results:
        for index in adDicts:
            if adDicts[index]['advice'] is not None: # Noneを含まない場合
                if adDicts[index]['advice'].find(kensaku[0]) != -1: # adviceに類似度の高い単語が含まれている場合
                    rateCount.append([adDicts[index]["reportNo"], adDicts[index]["companyName"], kensaku[1]]) # 類似度を用いて推薦機能を実装するための配列
                    reportNoType[adDicts[index]["reportNo"]] = adDicts[index]["companyType"]
                    #print(rateCount) # 出力例:[reportNo, companyName, 類似語の出現0・1]
                    cosSimilar[adDicts[index]["reportNo"]] = np.dot(adDicts[index]['vectorSum'], inputVectorSum) / (adDicts[index]['vectorLength'] * inputVectorLength) # 入力の文章と各文書ごとにコサイン類似度を計算

    reportDict = {} # 類似語を含むアドバイスの類似度をreport_no毎に足し算する
    # 同じ企業名で類似度を合計する
    fno1Comp = collections.Counter([comp[0] for comp in rateCount])
    rateCountNp = np.array(rateCount)

    compRecommendDic = {}
    simCosDic = {} # 報告書ごとの類似度の合計、cos類似度を格納する
    no_name = [] # report_no and company_name
    t = 0
    start = time.time()
    reportNp = rateCountNp[:, [0]].reshape(-1,)
    rateCountNp = rateCountNp[:, [1, 2]]
    for comp_no in fno1Comp:
        typeRate = 0
        # [企業のreport_no, report_noに含まれる類似語の数, 含まれている類似語の類似度全てを抽出]
        # 出現(0,1) + ((類似語出現回数- 1) * 0.05) * 類似度の合計
        similarSum = rateCountNp[np.where(reportNp == str(comp_no))]
        no_name.append([comp_no, similarSum[0][0]])
        simSum = np.sum(similarSum[:,1].reshape(-1,).astype(np.float64))
        if TYPE: # 業種を考慮した計算
            if reportNoType[comp_no] != input_comp_type or input_comp_type == None: # 選択されていない業種を低く設定する
                typeRate = 0.5
            else:
                typeRate = 1
        simLog = 0.0001 if math.log(simSum, 10) < 0 else math.log(simSum, 10)
        if ALGORITHMTYPE == 0:
            # type0: 類似語の合計 * 業種（メタ情報） * コサイン類似度
            compRecommendDic[comp_no] = simSum * typeRate * cosSimilar[comp_no]
        elif ALGORITHMTYPE == 1:
            # type1: log(類似語の合計) * 業種（メタ情報） * コサイン類似度
            compRecommendDic[comp_no] = simLog * typeRate * cosSimilar[comp_no]
        elif ALGORITHMTYPE == 2:
            # type2: log(類似語の合計) + 業種（メタ情報） + コサイン類似度
            compRecommendDic[comp_no] = simLog + typeRate + cosSimilar[comp_no]
        simCosDic[comp_no] = [simSum, simLog, typeRate, cosSimilar[comp_no]]
        timer = time.time() - start
        t += timer
        #print(timer)


    no_name = np.array(no_name)
    advice_json = {}
    print("類似度が", similaryty, "以上の類似単語を含む報告書を<式：出現(0,1) + ((類似語出現回数- 1) * 0.05) * 類似度の合計>にしたがって推薦度を計算した結果")
    print('順位 : 報告書No : 推薦度 : 企業名')
    for index, primaryComp in enumerate(sorted(compRecommendDic.items(), key=lambda x: x[1], reverse=True)[:20]):
        ranking = index + 1
        currentCompanyName = no_name[np.where(no_name[:, [0]].reshape(-1,) == str(primaryComp[0]))][0,[1]][0]
        recommend = str(ranking) + ',' + str(primaryComp[0]) + ',' + str(primaryComp[1]) + ',' + str(currentCompanyName) + ','
        print(recommend)
        advice_json[ranking] = primaryComp[0]
    print(json_dump(advice_json))

def json_dump(dictionary):
    import json
    return json.dumps(dictionary, sort_keys=True, indent=4)

def calc(equation):
    words = parser_mecab(equation)
    neighbor_word(words, inputText=equation)

if __name__=="__main__":
    input_comp_type = compTypeList.createList()
    ALGORITHMTYPE = compTypeList.choiceAlgorithm()
    equation = compTypeList.inputDoc()
    calc(equation)


