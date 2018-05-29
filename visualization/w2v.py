# -*- coding: utf-8 -*-

def draw_2d_2groups(vectors, target1, target2, topn=100):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager

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
    plt.figure(figsize=(9,9))
    plt.scatter(Y[:, 0], Y[:,1], color=colors)

    for w, x, y, c in zip(similars[:], Y[:, 0], Y[:,1], colors):
        plt.annotate(w, xy=(x, y), xytext=(3,3), textcoords='offset points', fontproperties=prop, fontsize=16, color=c)
    plt.show()

def get_pca(vectors, target, topn=40):
    from sklearn.decomposition import PCA
    import numpy as np

    similars = [w[0] for w in vectors.most_similar(target, topn=topn)]
    similars.insert(0, target)
    print(similars)
    similars = [vectors.wv[w] for w in similars]

    pca = PCA(n_components=2)
    pca.fit(similars)
    return pca.fit_transform(similars)

def pca_2d_2groups(vectors, target1, target2, topn=100):
    from matplotlib import pyplot as plt

    transformed1 = get_pca(vectors, target1)
    transformed2 = get_pca(vectors, target2)

    # 主成分をプロットする
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(transformed1[:, 0], transformed1[:, 1], c="red", label="high", marker="^")
    ax.scatter(transformed2[:, 0], transformed2[:, 1], c="green", label="low", marker="s")
    ax.set_title('2単語の類似度のベクトル次元削減')
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')

    ax.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    from gensim.models import word2vec

    FONTPATH = 'Osaka.ttc'
    MODEL_FILENAME = "../model/advice20180529.model"
    w2v = word2vec.Word2Vec.load(MODEL_FILENAME)

    draw_2d_2groups(w2v, '面接', 'グループディスカッション', topn=40)
    pca_2d_2groups(w2v, '面接', 'グループディスカッション', topn=40)


