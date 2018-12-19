# -*- coding: utf-8 -*-

from natto import MeCab
mc = MeCab('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

def parser_mecab(text):
    words = []
    for n in mc.parse(text, as_nodes=True):
        node = n.feature.split(',');
        if node[0] != '助詞' and node[0] != '助動詞' and node[0] != '記号' and node[1] != '数':
        #if node[0] != '助詞' and node[0] != '助動詞' and node[0] != '記号' and node[1] != '数' and node[0] != '動詞' and node[0] != '副詞':
            if node[0] == '動詞':
                words.append(node[6])
            elif node[0] == 'BOS/EOS':
                continue
            else:
                words.append(n.surface)
    return words

def pluck_ne(text):
    # 固有名詞の単語を配列で返す
    words = []
    for n in mc.parse(text, as_nodes=True):
        node = n.feature.split(',');
        if node[1] == '固有名詞':
            words.append(n.surface)
    return words

def is_noun(word):
    for n in mc.parse(word, as_nodes=True):
        node = n.feature.split(',');
        if node[0] == '名詞':
            return True
        else:
            return False

def parser_space(text):
    words = ""
    for n in mc.parse(text, as_nodes=True):
        node = n.feature.split(',');
        if node[0] != '助詞' and node[0] != '助動詞' and node[0] != '記号' and node[1] != '数':
            if node[0] == '動詞':
                words += node[6]
            elif node[0] == 'BOS/EOS':
                continue
            else:
                words += n.surface
        words += " "
    return words

def parser_juman(text):
    from pyknp import Jumanpp
    jumanpp = Jumanpp()

    result = jumanpp.analysis(text)
    words = []

    for n in result.mrph_list():
        if n.hinsi != '助詞' and n.hinsi != '助動詞' and n.hinsi != '特殊' and n.bunrui != "空白":
            if n.hinsi == '動詞':
                words.append(n.genkei)
            else:
                words.append(n.midasi)
    return words

#print(mrph.midasi, mrph.yomi, mrph.genkei, mrph.hinsi, mrph.bunrui, mrph.katuyou1, mrph.katuyou2, mrph.imis, mrph.repname)
