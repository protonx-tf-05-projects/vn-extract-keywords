import py_vncorenlp
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = []
with open('vietnamese-stopwords.txt', encoding='utf8') as f:
      for line in f:
            stop_words.append(line.strip())

doc = ''
with open('input_line.txt', encoding='utf8') as f:
      doc = f.read()


def removeStopWords(o_sen):
      words = [word for word in o_sen.split() if word not in stop_words]
      return " ".join(words)

n_gram_range = (2, 2)

py_vncorenlp.download_model(save_dir=os.path.abspath('./vncorenlp'))

# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.abspath('./vncorenlp'))

doc_segmented = rdrsegmenter.word_segment(doc)
# Extract candidate words/phrases

candidates = []
for r in [(1, 1), (2, 2)]:
      count = CountVectorizer(ngram_range=r).fit([removeStopWords(doc)])
      nr_candidates = count.get_feature_names()
      candidates.extend(nr_candidates)

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)


top_n = 20
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)

