from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle

with open('../advice.pickle', 'rb') as f:
    advice = pickle.load(f)

training_code = []
for reportNo in advice:
    training_code.append(
            TaggedDocument(words=advice[reportNo]['advice_divide_mecab'], tags=[str(advice[reportNo]['companyName'])])
    )

# dm=0 DBOW dm=1 dmpv
model = Doc2Vec(documents=training_code, size=100 , window=10, min_count=1, dm=1, iter=400, negative=15, sample=1e-6)

model.save('prototype.model')

