import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def load_data():
    return pd.read_pickle('processed_movies.pkl')

def predict_rating(features): #here features is a dictionary
    df = load_data() #loading the data
    mlb = MultiLabelBinarizer() #mlb actually converts list of labels -> one hot encoder
    # ['Action','Drama'] => [0,1,1,0]....
    #transforms the genre column into matrix (binary)

    genres_onehot = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)
    #genres onhot will be a 2d matrix like having labels as columns -> 1 present and 0 absent
    #creating X (independent set) by merging these columns with genres_onehot
    X = pd.concat([df[['revenue', 'popularity', 'runtime']], genres_onehot], axis=1)
    y = df['vote_average'] #dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting into train and test dataset
    model = LinearRegression() #initializing model simple linear regression model
    model.fit(X_train, y_train) #fittting the train data
    #converting the input genres into one hot -> as changed earlier it will return genre else []
    #transform method we have to use. as we already fitted the earlier data
    genres_input = pd.DataFrame(mlb.transform([features.get('genres', [])]), columns=mlb.classes_)
    # here get method used to fetch the respective data out of the dictionary else 0 is returned.
    input_df = pd.DataFrame([[features.get('revenue', 0), features.get('popularity', 0), features.get('runtime', 0)]], 
                            columns=['revenue', 'popularity', 'runtime'])
    #full df -> by concatenating
    input_full = pd.concat([input_df, genres_input], axis=1).reindex(columns=X.columns, fill_value=0)
    return model.predict(input_full)[0] #returns prediction

def get_trends(): #returns the trends 
    df = load_data()
    #uses groupby function it to group by year based on the revenue and mean
    trends = df.groupby('year')['revenue'].mean().reset_index()
    return trends.to_dict('records') #returns the trends as a form of dict

def get_correlations(): #getting the correlations based on the budget revenue popularity vote avg runtime
    df = load_data()
    corr = df[['budget', 'revenue', 'popularity', 'vote_average', 'runtime']].corr()
    return corr.to_dict() #returns the ans in the form of dictionary

def genre_popularity_over_time(): #getting the genre popularity over the time
    df = load_data()
    df_exploded = df.explode('genres') #it explodes like for a movie multiple genres
    #then it will convert the movie for each genre one row
    #like movie1 ["action",'drama]
    #after exploding it will be movie1 action and movie 1 drama 
    genre_trends = df_exploded.groupby(['year', 'genres'])['popularity'].mean().unstack().fillna(0) #unstack opens up the genres to be
    #the new columns and then replaces the NA values with 0
    return genre_trends

def extract_keywords_from_overview(movie_title, n=10):#n represents the keywords to return
    df = load_data() 
    overview = df[df['title'] == movie_title]['overview'].values[0] #extracts the overview from the df where movie title matches 
    stop_words = set(stopwords.words('english')) #here stopwords are used to remove unwanted words
    tokens = word_tokenize(overview.lower().translate(str.maketrans('', '', string.punctuation))) #same preprocessing: lower->remove punctuations (translate) -> tokenize
    freq = FreqDist([w for w in tokens if w not in stop_words]) #builds a freq distribution 
    #like ['king':3,'man':2...]
    return freq.most_common(n) #and returns the most common 

if __name__ == '__main__':
    print(predict_rating({'revenue': 100000000, 'popularity': 50, 'runtime': 120, 'genres': ['Action']}))
    print(get_correlations())
    print(extract_keywords_from_overview('Inception'))