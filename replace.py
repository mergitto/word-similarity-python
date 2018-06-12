# -*- coding: utf-8 -*-

def change_word(word):
    word = word.replace('it', 'ict')
    word = word.replace("ウェブ", "web")
    word = word.replace("gd", "グループディスカッション")
    word = word.replace("pg", "プログラマー")
    word = word.replace("openes", "エントリーシート")
    word = word.replace("es", "エントリーシート")
    word = word.replace("oes", "エントリーシート")
    word = word.replace("se", "システムエンジニア")
    return word

def decode_word(word):
    word = word.replace('ict', 'it')
    word = word.replace("web", "ウェブ")
    word = word.replace("グループディスカッション", "gd")
    word = word.replace("プログラマー", "pg")
    word = word.replace("エントリーシート", "es")
    word = word.replace("システムエンジニア", "se")
    return word
