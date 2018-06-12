import pickle

def createList(adDicts):
    typeList = []
    for t in adDicts.values():
        if t['companyType'] != None:
            typeList.append(t['companyType'])
    company_type = sorted(list(set(typeList)))
    company_type.append(None)
    return company_type


def inputDoc():
    while True:
        document = input("あなたが欲しい情報を20文字以上の文章で入力してください:")
        if len(document) > 20:
            return document

def choiceAlgorithm():
    while True:
        algorithmtype = int(input('アルゴリズムの種類を0~1で選んでください:'))
        if algorithmtype >= 0 and algorithmtype <= 1:
            return algorithmtype

def weightingSimilar(wordList):
    print('あなたが入力した文章の単語の中から最も重要な単語を以下から選び数字で選択してください')
    for index, word in enumerate(wordList):
        print(index, ':', word)
    selectWordIndex = input('重要な単語の数字:')
    return selectWordIndex

def evalMatch(currentCompanyName):
    while True:
        inp = input('入力と「'+ currentCompanyName +'」の報告書の一致度を1.0〜5.0で評価してください：')
        if isfloat(inp) and float(inp) <= 5.0 and float(inp) >= 1.0:
            break
        print('数値を1.0〜5.0で入力してください')
    return inp

def evalMeaning(currentCompanyName):
    while True:
        mean = input('入力に対して「'+ currentCompanyName +'」の報告書がどのぐらいためになったかを1.0〜5.0で評価してください：')
        if isfloat(mean) and float(mean) <= 5.0 and float(mean) >= 1.0:
            break
        print('数値を1.0〜5.0で入力してください')
    return mean

def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
