import pickle
from statistics import mean, median, stdev

with open('advice_10_tfidf.pickle', 'rb') as f:
    advices = pickle.load(f)

for advice in advices.values():
    advice_tfidf = advice['tfidf'].items()
    if len(advice_tfidf) >= 10:
        tfidf_max = max([(value, key) for key,value in advice_tfidf])
        tfidf_mean = median([value for key,value in advice_tfidf])
        if tfidf_mean > 0.3:
            print(advice['reportNo'], ":", tfidf_mean, tfidf_max, stdev([v for k,v in advice_tfidf]), len(advice_tfidf))
