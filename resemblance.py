# -*- coding: utf-8 -*-

import gensim
from gensim.models import word2vec
from gensim import corpora
import sys
import collections
import numpy as np
import re
import pickle
from natto import MeCab
import compTypeList
import math

mc = MeCab('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd') # ipadicの辞書を利用

#定数の宣言
similaryty = 0.50 # 類似度を設定する
INPUTWEIGHT = 1.0 # 入力文字の重み（仮想的な類似度）
PRIORITYRATE = 5 # 重要単語を選択した時に付加する類似語の類似度の倍率
LOWPRIORITYRATE = 0.5 # 非重要単語を選択した時に付加する類似語の類似度の倍率
INPUTWORDPATH = './output/inputWord.txt' # 入力単語を保存するファイル
RECCOMPPATH = './output/enterprise.csv' # 推薦企業の上位を保存するファイル
EVALUATIONPATH = './output/evaluation.csv' # 推薦企業の評価を保存するファイル
INPUTTYPEPATH = './output/inputType.csv' # 推薦企業の評価を保存するファイル
WRITE = True # 入力内容を書き込むか否か Trueなら書き込み、Falseなら書き込まない
WEIGHTING = True # 入力文字のから重要単語を選択する場合はTrue,しない場合はFalse
TYPE = True
############

model   = word2vec.Word2Vec.load(sys.argv[1])
# LDAによるトピック分類を利用した推薦のためのモデル読み込み
def loadLda(text=None):
    test_words = ""
    for n in mc.parse(text, as_nodes=True):
        node = n.feature.split(',');
        if node[0] != '助詞' and node[0] != '助動詞' and node[0] != '記号' and node[1] != '数':
            if node[0] == '動詞':
                test_words += node[6]
            else:
                test_words += n.surface
            test_words += " "
    # テスト用で適当な文章を作成し、どのトピックに当たるかを出力させてみる
    test_documents = [test_words]
    test_texts = [[word for word in document.split()] for document in test_documents]

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
            fileInput(po+' ', INPUTWORDPATH)
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
            fileInput(e.args[0]+' ', INPUTWORDPATH)
            print('「' + po + '」という単語は見つからなかったです')
            tmpWordCheck += '0,' + po + ','
        count += 1
    fileInput('\n', INPUTWORDPATH)
    inputVectorLength = np.linalg.norm(inputVectorSum)

    words = {'positive': posi, 'negative': nega}
    # adDictsのpickleをロードする
    with open('./advice_10.pickle', 'rb') as f: # トピック分類の情報を付加したデータをpickleでロード
        adDicts = pickle.load(f)
    rateCount = []
    topicDic = {} # 入力と文書ごとのトピック積和を格納
    cosSimilar = {} # 入力と文書ごとのコサイン類似度を格納
    reportNoType = {} # 報告書Noと業種の辞書
    for kensaku in results:
        for index in adDicts:
            if adDicts[index]['advice'] is not None: # Noneを含まない場合
                if adDicts[index]['advice'].lower().find(kensaku[0]) != -1: # adviceに類似度の高い単語が含まれている場合
                    adDicts[index]['companyName'] = adDicts[index]["companyName"].replace("\u3000", " ") # 全角空白を半角空白に置換
                    rateCount.append([adDicts[index]["reportNo"], adDicts[index]["companyName"], kensaku[1]]) # 類似度を用いて推薦機能を実装するための配列
                    reportNoType[adDicts[index]["reportNo"]] = adDicts[index]["companyType"]
                    topicSum = 0
                    topicDic[adDicts[index]["reportNo"]] = topicSum #  topicの掛け合わせ値と
                    #print(rateCount) # 出力例:[reportNo, companyName, 類似語の出現0・1]
                    cosSimilar[adDicts[index]["reportNo"]] = np.dot(adDicts[index]['vectorSum'], inputVectorSum) / (adDicts[index]['vectorLength'] * inputVectorLength) # 入力の文章と各文書ごとにコサイン類似度を計算

    #print("類似度が", similaryty, "以上の単語を含んでいた場合に、index・企業名・類似度・入力文字から得た報告書リストに含んでいるか否かを表したリストを表示")
    reportDict = {} # 類似語を含むアドバイスの類似度をreport_no毎に足し算する
    # 同じ企業名で類似度を合計する
    fno1Comp = collections.Counter([comp[0] for comp in rateCount])
    rateCountNp = np.array(rateCount)

    compRecommendDic = {}
    simCosDic = {} # 報告書ごとの類似度の合計、cos類似度を格納する
    no_name = [] # report_no and company_name
    for comp_no in fno1Comp:
        typeRate = 0 # 業種のレート、アルゴリズムに考慮する
        # [企業のreport_no, report_noに含まれる類似語の数, 含まれている類似語の類似度全てを抽出]
        # キーワードの出現回数を考慮した推薦のための式
        # 出現(0,1) + ((類似語出現回数- 1) * 0.05) * 類似度の合計
        similarSum = rateCountNp[np.where(rateCountNp[:, [0]].reshape(-1,) == str(comp_no))][:,[1, 2]]
        no_name.append([comp_no, similarSum[0][0]])
        if TYPE: # 業種を考慮した計算
            if reportNoType[comp_no] != input_comp_type or input_comp_type == None: # 選択されていない業種を低く設定する
                typeRate = 0.5
            else:
                typeRate = 1
            #typeRate = 1 if reportNoType[comp_no] == input_comp_type else typeRate = 0.3
        #compRecommendDic[comp_no] = normSimRepo[comp_no] * typeRate * cosSimilar[comp_no] # 類似語出現回数 * 類似語の合計 * 業種（メタ情報） * コサイン類似度（入力と文章の類似度）
        simSum = sum(similarSum[:,1].reshape(-1,).astype(np.float64))
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
    inputTypeTmp = str(input_comp_type) + ',' + str(ALGORITHMTYPE) + ',' + str(equation) + '\n'
    fileInput(inputTypeTmp, INPUTTYPEPATH)
    if ALGORITHMTYPE == 0:
        fileInput(tmpWordCheck + ',' + posi[int(weightingFlag)] + '\n', RECCOMPPATH)
    else:
        fileInput(tmpWordCheck + '\n', RECCOMPPATH)
    no_name = np.array(no_name)
    print("類似度が", similaryty, "以上の類似単語を含む報告書を<式：出現(0,1) + ((類似語出現回数- 1) * 0.05) * 類似度の合計>にしたがって推薦度を計算した結果")
    print('順位 : 報告書No : 推薦度 : 企業名')
    for index, primaryComp in enumerate(sorted(compRecommendDic.items(), key=lambda x: x[1], reverse=True)[:10]):
        currentCompanyName = no_name[np.where(no_name[:, [0]].reshape(-1,) == str(primaryComp[0]))][0,[1]][0]
        recommend = str(index+1) + ',' + str(primaryComp[0]) + ',' + str(primaryComp[1]) + ',' + str(currentCompanyName) + ','
        print(recommend)
        fileInput(recommend+'\n', RECCOMPPATH)
    fileInput('\n', RECCOMPPATH)
    print("\n")
    if WRITE == True:
        for index, primaryComp in enumerate(sorted(compRecommendDic.items(), key=lambda x: x[1], reverse=True)[:5]):
            currentCompanyName = no_name[np.where(no_name[:, [0]].reshape(-1,) == str(primaryComp[0]))][0,[1]][0]
            recommend = str(index+1) + ',' + str(primaryComp[0]) + ',' + str(primaryComp[1]) + ',' + str(currentCompanyName)
            print('報告書No : ', str(primaryComp[0]))
            print('企 業 名 : ', str(currentCompanyName))
            inp = compTypeList.evalMatch(currentCompanyName) # 報告書と入力の一致度を評価
            meaning = compTypeList.evalMeaning(currentCompanyName) # 報告書がためになったかを評価
            print("\n")
            tmpEvaluation = str(recommend) + "," + str(inp) + "," + str(meaning) + "," + str(inputTypeTmp)
            simMethaCosCSV = ''
            for calcList in simCosDic[primaryComp[0]]:
                simMethaCosCSV += ',' +str(calcList)
            tmpEvaluation = tmpEvaluation.replace("\n", "") + simMethaCosCSV[:-1] + '\n'
            fileInput(tmpEvaluation, EVALUATIONPATH)

def calc(equation):
    words = []
    for n in mc.parse(equation, as_nodes=True):
        node = n.feature.split(',')
        if node[0] != '助詞' and node[0] != '助動詞' and node[0] != '記号' and node[1] != '数' and node[0] != '動詞' and node[0] != '副詞':
            if node[0] == '動詞':
                words.append(node[6])
            elif node[0] == 'BOS/EOS':
                continue
            else:
                words.append(n.surface)
    neighbor_word(words, inputText=equation)
    fileInput('\n', EVALUATIONPATH)

def fileInput(word, filepath):
    if WRITE == True:
        file = open(filepath, 'a')  #追加書き込みモードでオープン
        file.writelines(word)

if __name__=="__main__":
    input_comp_type = compTypeList.createList()
    ALGORITHMTYPE = compTypeList.choiceAlgorithm()
    equation = compTypeList.inputDoc()
    calc(equation)


