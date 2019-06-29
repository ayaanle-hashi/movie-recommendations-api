import pandas as pd
import warnings
from flask import Flask
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

warnings.filterwarnings("ignore")

df = pd.read_csv("https://docs.google.com/spreadsheets/d/1kyDvR6N8Bed0zakWyGlTx-1Ix3IL3Bi-nKY6tm5rDaY/export?format=csv&id=1kyDvR6N8Bed0zakWyGlTx-1Ix3IL3Bi-nKY6tm5rDaY&gid=1711541452")

tf = TfidfVectorizer(stop_words='english',ngram_range=(1,2),analyzer='word')

tfidf_matrix = tf.fit_transform(df['Plot'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

titles = df['Title']

indices =pd.Series(df.index,index=df['Title'])

@app.route('/recommendations/<title>',methods=['GET'])
def make_recomendataions(title):
  '''
  Function that finds the top 10 most similar movies based on cosine similarity between tfidf vector
  '''
  try:
    idx =indices[title]
    similarity_score = list(enumerate(cosine_sim[idx]))
    similarity_score = sorted(similarity_score,key=lambda x: x[1],reverse=True)
    similarity_score = similarity_score [1:11]
    movie_indices = [i[0] for i in similarity_score]
    recommendations = list(titles.iloc[movie_indices])
    data = pd.DataFrame(data=recommendations)
    return (data.to_string(index=False,header=False))

  except KeyError as e:
    return "ERROR MOVIE NOT FOUND",404
    
if __name__ == "__main__":
  app.run(debug=True)