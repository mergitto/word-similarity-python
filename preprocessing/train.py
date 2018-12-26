################
# 分散表現モデル作成
# [参照]
# https://qiita.com/mergit/items/822dc49343c887019d44
# [使用方法]
# python train.py 分かち書きしたテキスト.txt 適当なモデル名.model
################

from gensim.models import word2vec
import sys

sentences = word2vec.LineSentence(sys.argv[1])
model = word2vec.Word2Vec(sentences,
                          sg=1, # 0 = C-BOW, 1 = skip-gram
                          size=100, # ベクトルの次元数
                          min_count=1, # 出現頻度がmin_count未満の単語削除
                          window=10, # 学習する前後の単語数
                          hs=0, # 階層的ソフトマックス関数の使用有無 0 = 使用しない, 1 = 使用する
                          iter=200, # 学習回数
                          negative=10 # ネガティブサンプリングに用いる単語数
                          )
model.save(sys.argv[2])
