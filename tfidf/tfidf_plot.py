import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_pickle():
    with open('corpus_tfidf.pickle', 'rb') as f:
        return pickle.load(f)

def save_image(save_file_path):
    plt.savefig(save_file_path)

def to_hist(tfidf_list, save_file_path):
    plt.hist(tfidf_list)
    save_image(save_file_path)
    #plt.show()
    plt.clf()

def to_plot(tfidf_list, save_file_path):
    tfidf_list = sorted(tfidf_list, reverse=True)
    plt.plot(tfidf_list)
    save_image(save_file_path)
    #plt.show()
    plt.clf()

def to_sns_plot(tfidf_list, save_file_path):
    sns.distplot(tfidf_list)
    save_image(save_file_path)
    plt.show()
    plt.clf()

if __name__ == '__main__':
    tfidf_model = load_pickle()

    tfidf_values = [round(word_tfidf[1], 3) for each_corpus in tfidf_model for word_tfidf in each_corpus]
    to_hist(tfidf_values, '/Users/riki/Desktop/tfidf_hist.pdf')
    to_plot(tfidf_values, '/Users/riki/Desktop/tfidf_plot.pdf')


