## 文書と入力からを様々な類似度を計算する
word2vecなどで単語同士の類似度をコサイン類似度から計算するが、文書や単語同士の類似度の計算には他にも存在する

- Jaccard係数
- Dice係数
- Simpson係数

などが挙げられる

## 手順
1. `cp ./const-sample.py ./const.py`
1. const.pyの設定をする（postgresqlとsql,LDA学習のために利用する分かち書き文書）
1. `mkdir pickle` pickle用
1. `python pgsql.py` データベースから報告書の情報をpickleで出力する
1. `mkdir model` 分散表現学習モデル用
1. `python batch.py` 前処理でデータを追加する処理を実行

