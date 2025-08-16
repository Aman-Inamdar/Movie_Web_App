import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# def preprocess_text(text):
#     #stop_words = set(stopwords.words('english')) #stopwords to remove the words like is,the,am...
#     #lowering all the text to maintain to uniformity followed by it 
#     #makes a translation table which removes the all punctuation signs
#     #string.punctuation: contains all the punctuation
#     text = text.lower().translate(str.maketrans('', '', string.punctuation))

#     #tokenizing the all the strings to get a list ["a","b"...]

#     tokens = word_tokenize(text)

#     #returns the joined string from the tokens(list of words) by removing stopwords
#     return ' '.join([word for word in tokens if word not in stop_words])

class MovieRecommender:
    #constructor
    def __init__(self): #whenever the class is created the constructor created and loads
        #necessary infos..
        self.df = pd.read_pickle('processed_movies.pkl') #Loads the preprocessed movies file 
        #soup represents the single string created by joining all the words- vertically we are joining

        self.df['soup'] = self.df.apply(self.create_soup, axis=1)
        #vectorizing using TFID(Term frequncy Inverse)
        self.vectorizer = TfidfVectorizer(stop_words='english',max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['soup'])

    #creates a soup column with all the info combined
    def create_soup(self, row):
        #for row it joins all the genres which we actually parsed them into the lists of strings
        genres = ' '.join(row['genres'])
        companies = ' '.join(row['production_companies'])
        lang = row['original_language']
        overview = row['overview'].lower()
        return f"{genres} {companies} {lang} {overview}"
    
    #recommends the movies based on the cosine simillarity - Content based recommendation
    def get_recommendations(self, title, n=10): #default recommendation is 10
        if title not in self.df['title'].values:
            return [] #checks for the movie title in the all the movies
        idx = self.df[self.df['title'] == title].index[0]
        movie_vec = self.tfidf_matrix[idx]
        # Compute similarity (O(N))
        sim_scores = cosine_similarity(movie_vec, self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.df.iloc[movie_indices][['title', 'vote_average', 'popularity', 'year']].to_dict('records')


if __name__=='__main__':
    rec=MovieRecommender()
    print(rec.get_recommendations('Titanic',n=5))    