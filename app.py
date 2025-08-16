import streamlit as st
from recommender import MovieRecommender
from analysis import predict_rating, get_trends, get_correlations, extract_keywords_from_overview, genre_popularity_over_time
from viz import plot_revenue_trends, plot_correlation_heatmap, plot_genre_popularity
import pandas as pd

# Cache recommender
@st.cache_resource
def load_recommender():
    return MovieRecommender()

# Load once
try:
    rec = load_recommender()
    movies_list = rec.df['title'].sort_values().tolist()
except Exception as e:
    st.error(f"Failed to load recommender: {e}")
    st.stop()

# App config
st.set_page_config(page_title="Movie Explorer", layout="wide")
page = st.sidebar.selectbox("Navigate", ["Home", "Recommender", "Analytics"])

if page == "Home":
    st.title("Welcome to Movie Explorer App")
    st.write("Explore TMDB movies: Recommend, Analyze, Visualize!")
    st.write(f"Loaded {len(rec.df)} movies.")
    st.write("Use the sidebar to navigate.")

elif page == "Recommender":
    st.title("Movie Recommender")
    search = st.text_input("Search Movie (type to filter)")
    filtered_movies = [m for m in movies_list if search.lower() in m.lower()] if search else movies_list
    selected_movie = st.selectbox("Pick a Movie", filtered_movies)
    if st.button("Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            recs = rec.get_recommendations(selected_movie, n=10)
        if recs:
            st.subheader(f"Top 10 Similar to {selected_movie}")
            st.table(pd.DataFrame(recs))
            
            # Details
            movie_data = rec.df[rec.df['title'] == selected_movie].iloc[0]
            st.subheader("Movie Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Popularity: {movie_data['popularity']:.2f}")
                st.write(f"Revenue: ${movie_data['revenue']:,}")
                st.write(f"Runtime: {movie_data['runtime']} min")
            with col2:
                st.write(f"Vote Average: {movie_data['vote_average']}/10")
                st.write(f"Genres: {', '.join(movie_data['genres'])}")
                st.write(f"Companies: {', '.join(movie_data['production_companies'])}")
            
            # Poster
            if 'poster_path' in movie_data and movie_data['poster_path'] and not pd.isna(movie_data['poster_path']):
                st.image(f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}", width=200)
            
            # NLP
            keywords = extract_keywords_from_overview(selected_movie)
            st.subheader("Overview Keywords")
            st.write([kw[0] for kw in keywords])
        else:
            st.error("Movie not found. Try another.")

elif page == "Analytics":
    st.title("Movie Analytics")
    
    # Trends
    trends = get_trends()
    st.subheader("Revenue Trends Over Time")
    st.pyplot(plot_revenue_trends(trends))
    
    # Correlations
    corrs = get_correlations()
    st.subheader("Feature Correlations")
    st.pyplot(plot_correlation_heatmap(corrs))
    
    # Genres
    genre_data = genre_popularity_over_time()
    st.subheader("Genre Popularity Over Time")
    st.pyplot(plot_genre_popularity(genre_data))
    
    # Predict
    st.subheader("Predict Movie Rating")
    revenue = st.number_input("Revenue ($)", 0, 1000000000, 100000000)
    popularity = st.number_input("Popularity", 0.0, 1000.0, 50.0)
    runtime = st.number_input("Runtime (min)", 0, 300, 120)
    genres = st.multiselect("Genres", ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'])
    if st.button("Predict"):
        pred = predict_rating({'revenue': revenue, 'popularity': popularity, 'runtime': runtime, 'genres': genres})
        st.write(f"Predicted Rating: {pred:.2f}/10")