## doc2vecで文書間の類似度を求める

## 手順
1. {'タグ名', '文章'}のような辞書を作成しておく(今回は前もってpickle化しておく)
1. (ここの設定を自分で変更)[https://github.com/mergitto/word-similarity-python/blob/master/doc2vec/doc2vec.py#L11]
1. `python doc2vec.py`
1. 学習したモデルを利用して類似度を出す
  1.  `python similar.py モデル.model "タグ名"`
