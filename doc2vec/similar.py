from gensim.models import Doc2Vec
import sys

model = Doc2Vec.load(sys.argv[1])

def search_similar_texts(words):
    x = model.infer_vector(words)
    most_similar_texts = model.docvecs.most_similar([x])
    for similar_text in most_similar_texts:
        print(similar_text[0])

for i in model.docvecs.most_similar([sys.argv[2]]):
    print(i)

#search_similar_texts(sys.argv[3])

