from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

FONTPATH = 'Osaka.ttc'

def draw_2d_2groups(vectors, target1, target2, topn=100):
    similars1 = [w[0] for w in vectors.most_similar(target1, topn=topn)]
    similars1.insert(0, target1)

    similars2 = [w[0] for w in vectors.most_similar(target2, topn=topn)]
    similars2.insert(0, target2)

    print(similars1)
    print(similars2)

    similars = similars1 + similars2

    colors = ['b']+['g']*(topn)+ ['r']+['orange']*(topn+1)

    X = [vectors.wv[w] for w in similars]
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(X)

    prop = font_manager.FontProperties(fname=FONTPATH)
    plt.figure(figsize=(20,20))
    plt.scatter(Y[:, 0], Y[:,1], color=colors)

    for w, x, y, c in zip(similars[:], Y[:, 0], Y[:,1], colors):
        plt.annotate(w, xy=(x, y), xytext=(3,3), textcoords='offset points', fontproperties=prop, fontsize=16, color=c)
    plt.show()


from gensim.models import word2vec
MODEL_FILENAME = "../model/advice20180529.model"
w2v = word2vec.Word2Vec.load(MODEL_FILENAME)


#vec = w2v['Java']
## ベクトルの空間数
#vec.shape
## ベクトルの値を表示
#print(str(vec)[:98]+'...')
#[-0.32416266 -0.06502125 -0.45745352 -0.0723946  -1.0601064   0.19136839   0.24223328  0.96732855 ...

#print(w2v.most_similar(positive=['spi'], topn=10))

draw_2d_2groups(w2v, '面接', 'グループディスカッション', topn=30)


