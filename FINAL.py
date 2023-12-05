import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from sklearn.impute import KNNImputer

from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer

df = pd.read_csv("/Users/amandarajda/desktop/final_df.csv")

df.info()
df.shape
df.head()

def get_index(Name):
    return df[df.Name == Name].index[0]
def get_title_from_index(index):
    return df[df.index == index]["Name"].values[0]
df['Description'].fillna('')

def get_recommendations_2(x):

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['Description'])
    cosine_sim = cosine_similarity(count_matrix)
    perfume_index = get_index(x)
    similar_perfumes = list(enumerate(cosine_sim[perfume_index]))
    sorted_similar = sorted(similar_perfumes,key=lambda x:x[1],reverse=True)[1:]

    i=0
    print("Top 5 similar Perfumes to "+x+" are:\n")
    for element in sorted_similar:
        print(get_title_from_index(element[0]), f"  - Similiarity score: ", format(sorted_similar[i][1], ".4f"))
        i=i+1
        if i>=5:
            break

get_recommendations_2("Yves Saint Laurent Libre")
get_recommendations_2("Yves Saint Laurent Black Opium")
get_recommendations_2("Burberry London")