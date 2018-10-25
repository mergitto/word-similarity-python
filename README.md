## 文書と入力からを様々な類似度を計算する
word2vecなどで単語同士の類似度をコサイン類似度から計算するが、文書や単語同士の類似度の計算には他にも存在する

- Jaccard係数
- Dice係数
- Simpson係数

などが挙げられる

## 手順
1. `cp ./const-sample.py ./const.py`
1. const.pyの設定をする（postgresqlとsql,LDA学習のために利用する分かち書き文書）
1. `python pgsql.py` データベースから報告書の情報をpickleで出力する
1. `mkdir model` 分散表現学習モデル用
1. addTopic.pyを`train = True`に設定する
1. `mkdir ldaModel` LDAモデル格納用
1. `python addTopic.py model/分散表現学習モデル名` LDAモデルの学習をする
1. addTopic.pyを`train = False`に設定する
1. `python addTopic.py model/分散表現学習モデル名` 学習したLDAモデルを`python pgsql.py`によって作成した辞書に追加する

