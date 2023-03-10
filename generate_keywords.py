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

py_vncorenlp.download_model(save_dir=os.path.abspath('./vncorenlp'))

# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.abspath('./vncorenlp'))

doc_segmented = rdrsegmenter.word_segment(doc)
# Extract candidate words/phrases

count = CountVectorizer(ngram_range=(1,1)).fit([removeStopWords(doc_segmented[0])])
candidates = count.get_feature_names()

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)


top_n = 10
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)