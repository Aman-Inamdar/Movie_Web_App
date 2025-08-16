Movie Explorer App
A Streamlit-based web application for exploring movies, providing personalized recommendations, detailed movie information, and analytics. Built for users in regions with TMDB API restrictions (e.g., India), it uses a local TMDB dataset (tmdb_movies.csv) for metadata and posters, OMDb API for actors/directors, and YouTube Data API v3 for embedded trailers.
Features

Recommender: Search or select a movie to view a details page with:
Poster, metadata (vote average, popularity, revenue, budget, runtime, genres, companies, tagline, status).
Actors and director (via OMDb API).
Embedded YouTube trailer (via YouTube Data API v3).
Sentiment score (based on overview).
10 recommended movies with posters and "Get Details" buttons to view their details.
"Add to Watchlist" and "Back to Original Movie" options.


Analytics:
Revenue trends over time (Plotly chart).
Feature correlations (heatmap).
Top movies by genre/year, sorted by popularity or rating.
Runtime impact on popularity/revenue.
Actor search (e.g., "Tom Hanks") with fuzzy title matching.
Predicted movie rating based on revenue, popularity, runtime, genres.


Watchlist: Save movies, view table, export to CSV, or clear.
Performance: Loads in 5–10s, recommendations in <1s, memory-efficient (5k–10k movies).

Prerequisites

Python: 3.8+
APIs:
OMDb API key: Sign up at omdbapi.com (optional, for actors/directors).
YouTube Data API v3 key: Create at console.cloud.google.com (for trailers).


Dataset: tmdb_movies.csv (TMDB movie data, included).
VPN/DNS: For poster access in India (e.g., ProtonVPN or DNS 8.8.8.8).

Setup

Clone Repository:
git clone https://github.com/your_username/movie-explorer-app.git
cd movie-explorer-app


Set Up Virtual Environment:
python -m venv venv
.\venv\Scripts\activate  # Windows


Install Dependencies:
pip install -r requirements.txt

Contents of requirements.txt:
streamlit
pandas
numpy
scikit-learn
nltk
plotly
requests
fuzzywuzzy
python-Levenshtein
streamlit-extras


Download NLTK Data:
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')


Add API Keys:Create .streamlit/secrets.toml:
OMDB_API_KEY = "your_omdb_key_here"
YOUTUBE_API_KEY = "your_youtube_api_key_here"


Prepare Data:Run data_prep.py to process tmdb_movies.csv into processed_movies.pkl (~5k–10k rows):
python data_prep.py



Usage

Run the App:
streamlit run app.py

Opens localhost:8501 in your browser.

Navigate:

Home: Overview and instructions.
Recommender: Search/select a movie (e.g., "Inception"). View details (poster, metadata, actors, trailer, sentiment). Click "Get Details" on recommended movies to explore further. Add to watchlist.
Analytics: View trends, correlations, top movies, runtime impact, actor search (e.g., "Tom Hanks"), or predict ratings.
Watchlist: View saved movies, export to CSV, or clear.


Troubleshooting:

Posters/Trailers Fail: Use VPN or DNS (8.8.8.8). Check API keys in secrets.toml.
Actor Search Fails: Use exact names (e.g., "Leonardo DiCaprio"). Test OMDb: curl "http://www.omdbapi.com/?s=Tom+Hanks&type=movie&apikey=your_key".
"Get Details" Fails: Check console (Ctrl+Shift+J). Add debug print in app.py:if st.button("Get Details", key=f"rec_detail_{title}_{i}"):
    st.write(f"Debug: Clicking Get Details for {title}")
    st.session_state['current_movie'] = title
    st.rerun()


Slow Performance: Run data_prep.py, check filtered shape (~5k–10k rows). Adjust vote_count > 300 if needed.



Deployment

Push to GitHub:
git add .
git commit -m "Initial commit"
git push origin main


Deploy to Streamlit Cloud:

Create account at streamlit.io.
Connect GitHub repo, select main branch, app.py as main file.
Upload processed_movies.pkl, watchlist.pkl.
Add secrets.toml with API keys in Streamlit Cloud settings.
Deploy (~5 mins).



Project Structure

app.py: Main Streamlit app with UI and logic.
recommender.py: Movie recommendation engine (TF-IDF, cosine similarity, OMDb/YouTube APIs).
analysis.py: Analytics functions (trends, correlations, actor search, etc.).
viz.py: Plotly visualizations.
data_prep.py: Processes tmdb_movies.csv into processed_movies.pkl.
tmdb_movies.csv: TMDB dataset.
processed_movies.pkl: Preprocessed movie data.
watchlist.pkl: Saved watchlist.
requirements.txt: Dependencies.
.streamlit/secrets.toml: API keys.

Known Issues

TMDB Restrictions: Posters may require VPN/DNS in India. Placeholders used if unavailable.
OMDb API: Limited requests without key. Actor/director data may be unavailable.
YouTube API: Quota limits apply. Fallback trailer ID used if request fails.

Future Enhancements

Hybrid Recommendations: Add user ratings page with collaborative filtering (e.g., surprise library).
Faster Search: Use Faiss for O(log N) similarity search.
UI Improvements: Add carousel for recommendations (streamlit-carousel) or feedback buttons.

Contributing
Submit issues or pull requests to GitHub repo. Ensure code follows PEP 8 and includes tests.
License
MIT License
