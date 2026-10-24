# importing pandas
import pandas as pd


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

credits=credits[['movie_id','cast','crew']]
credits.shape


movies = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)
movies = movies.drop("movie_id", axis=1)



movies=movies[['id','title','overview','genres','keywords','cast','crew']]


movies.dropna(inplace=True)
movies.shape

#imporint ast module to convert string to list
import ast
def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L


movies['genres']=movies['genres'].apply(convert)

movies['keywords']=movies['keywords'].apply(convert)


def convert3(obj):
    
    L=[]
    count=0
    for i in ast.literal_eval(obj):
        if count!=3:
            L.append(i['name'])
            count+=1
        else:
            break
    return L


movies['cast']=movies['cast'].apply(convert3)



def fetch_director(obj):
    L=[]

    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


movies['crew']=movies['crew'].apply(fetch_director)


movies["overview"] = movies["overview"].apply(lambda x: x.split())

print(movies['overview'][0])


movies['genres']=movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords']=movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew']=movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast']=movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])




movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']



new_df=movies[['id','title','tags']]


new_df.rename(columns={'id': 'movie_id'}, inplace=True)
print(new_df['tags'][0])


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))


new_df['tags']=new_df['tags'].apply(lambda x: x.lower())

import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


new_df['tags']=new_df['tags'].apply(stem)


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')


vectors = cv.fit_transform(new_df['tags']).toarray()
print(vectors.shape)

from sklearn.metrics.pairwise import cosine_similarity 