# -*- coding: utf-8 -*-

import pickle
from natto import MeCab
import sys
from calc import Calc

mc = MeCab('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

with open('./advice_10.pickle', 'rb') as f:
    advice = pickle.load(f)

def parser(text):
    words = []
    for n in mc.parse(text, as_nodes=True):
        node = n.feature.split(',');
        if node[0] != '助詞' and node[0] != '助動詞' and node[0] != '記号' and node[1] != '数':
        #if node[0] != '助詞' and node[0] != '助動詞' and node[0] != '記号' and node[1] != '数' and node[0] != '動詞' and node[0] != '副詞':
            if node[0] == '動詞':
                words.append(node[6])
            else:
                words.append(n.surface)
    return words

if __name__ == '__main__':
    parse_sentence = parser(sys.argv[1])

    for params in advice.values():
        if params['advice'] is None:
            continue
        parse_advice = parser(params['advice'])
        calculation = Calc()
        print("報告書No: " + str(params['reportNo']) + " 企業名" + params['companyName'])
        print("Jaccard: " + str(calculation.jaccard(parse_sentence, parse_advice)))
        print("Dice:" + str(calculation.dice(parse_sentence, parse_advice)))
        print("Simpson:" + str(calculation.simpson(parse_sentence, parse_advice)))
        print("\n")

