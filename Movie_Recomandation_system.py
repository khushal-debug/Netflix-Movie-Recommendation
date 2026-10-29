import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("netflix_titles.csv.zip")
print('head' ,df.head())
# from ydata_profiling import ProfileReport
# profile = ProfileReport(df, title="Netflix Data Profiling Report")
# profile.to_file("netflix_data_profiling_report.html")
print('info' ,df.info())
print('describe' ,df.describe())
print('columns' ,df.columns)
print('isnull' ,df.isnull().sum())
print('type value counts' ,df['type'].value_counts())
df['type'] = df['type'].map({'Movie': 1, 'TV Show': 0})
print('type value counts after mapping' ,df['type'].value_counts())
text_cols = ['director', 'cast', 'country', 'date_added', 'rating']
df[text_cols] = df[text_cols].fillna(' ')
print('null values after filling' ,df.isnull().sum())
features = ['director', 'cast', 'country', 'listed_in', 'description']
df['com_features'] = df[features].fillna('').agg(' '.join, axis=1)
print('combined features' ,df['com_features'].head())
from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['com_features'])
print('tfidf matrix shape' ,tfidf_matrix.shape)
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix )
print('cosine similarity shape' ,cosine_sim.shape)
indices = pd.Series(df.index, index=df['title']).drop_duplicates() 
def rec(title, n=5):
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        return "Movie not found "
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
print('recommendations for "Breaking Bad"' ,rec('Breaking Bad', 5))
print(df[df['title'].str.contains('Norm of the North', case=False)]['title'])
print(rec("Inception", 5 ))
print(rec("The Dark Knight", 5 ))
import streamlit as st
st.title("Netflix Movie Recommendation System")
title = st.text_input("Enter a movie title: You can try 'Breaking Bad', 'Inception', 'The Dark Knight', 'Norm of the North' etc.")
if st.button("Recommend"):
    results = rec(title, 5)
    if isinstance(results, str):
        st.error(results)
    else:
        st.subheader("Recommended Movies:")
        for movie in results:
            st.write(movie) 
