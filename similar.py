# -*- coding: utf-8 -*-

import pickle
from natto import MeCab
import sys
from calc import Calc
import time
from parse import *

SHOWNUM = 20

with open('./tfidf/advice_10_tfidf.pickle', 'rb') as f:
    advice = pickle.load(f)

if __name__ == '__main__':
    parse_sentence = parser_mecab(sys.argv[1])
    similarytyDict = {}

    for params in advice.values():
        parse_advice = params['advice_divide_mecab']

        calculation = Calc()
        jaccard_num = float(calculation.jaccard(parse_sentence, parse_advice))
        dice_num = float(calculation.dice(parse_sentence, parse_advice))
        simpson_num = float(calculation.simpson(parse_sentence, parse_advice)) if len(parse_advice) > 5 else 0

        similarytyDict[params['reportNo']] = {'companyName': params['companyName'], 'similar': {'jaccard': jaccard_num, 'dice': dice_num,'simpson': simpson_num}}

    top = sorted(similarytyDict.values(), key=lambda x: x['similar']['jaccard'], reverse=True)[:SHOWNUM]
    print('Jaccard')
    for index, t in enumerate(top):
        print(str(index+1) + ':' + t['companyName']+ ':' + str(t['similar']['jaccard']))

    top = sorted(similarytyDict.values(), key=lambda x: x['similar']['dice'], reverse=True)[:SHOWNUM]
    print('\nDice')
    for index, t in enumerate(top):
        print(str(index+1) + ':' + t['companyName']+ ':' + str(t['similar']['dice']))

    top = sorted(similarytyDict.values(), key=lambda x: x['similar']['simpson'], reverse=True)[:SHOWNUM]
    print('\nSimpson')
    for index, t in enumerate(top):
        print(str(index+1) + ':' + t['companyName']+ ':' + str(t['similar']['simpson']))
