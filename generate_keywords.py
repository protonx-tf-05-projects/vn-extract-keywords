import py_vncorenlp
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

n_gram_range = (1, 1)
stop_words = "english"


# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
py_vncorenlp.download_model(save_dir=os.path.abspath('./vncorenlp'))

# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.abspath('./vncorenlp'))

doc = """
         Bánh cheesecake ngọt mềm cực kỳ hấp dẫn và kích thích vị giác. Cùng học cách làm bánh cheesecake việt quất béo ngậy thơm ngon dễ làm tại nhà ngay nhé
      """

doc_segmented = rdrsegmenter.word_segment(doc)
# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
candidates = count.get_feature_names()


model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)


top_n = 10
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)

