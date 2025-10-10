import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

# NOTE: Detta är en fungerande lösning från tidigare uppgift.

class HybridRecommender:
    def __init__(self, movie_data_path, rating_data_path):
        self.movies_df = pd.read_csv(movie_data_path)
        self.ratings_df = pd.read_csv(rating_data_path)
        self.content_recommender = self._ContentBasedRecommender(self.movies_df)
        
        self.nmf_model = None; self.user_item_matrix = None; self.user_mapper = None
        self.movie_mapper = None; self.movie_inv_mapper = None

    # ta emot train_df, för att  itsället träna med träningsdata
    def fit(self, train_df):
        self.ratings_df = train_df
        self.content_recommender.fit()
        self.user_item_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        self.user_mapper = {uid: i for i, uid in enumerate(self.user_item_matrix.index)}
        self.movie_mapper = {mid: i for i, mid in enumerate(self.user_item_matrix.columns)}
        self.nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=500)
        self.nmf_model.fit(self.user_item_matrix)

    def recommend(self, user_id, movie_title_seed, num_recommendations=10):
        content_recs = self.content_recommender.recommend(movie_title_seed, num_recommendations)
        collaborative_recs = []
        if user_id in self.user_mapper:
            user_idx = self.user_mapper[user_id]
            user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
            user_P = self.nmf_model.transform(user_vector)
            item_Q = self.nmf_model.components_
            predicted_scores = np.dot(user_P, item_Q).flatten()
            scores_series = pd.Series(predicted_scores, index=self.user_item_matrix.columns)
            rated_movies = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId']
            scores_series = scores_series.drop(index=rated_movies, errors='ignore')
            top_movie_ids = scores_series.nlargest(num_recommendations).index.tolist()
            collaborative_recs = self.movies_df[self.movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()
        combined_recs = collaborative_recs + content_recs
        unique_recs = list(dict.fromkeys(combined_recs).keys())
        return unique_recs[:num_recommendations]

    class _ContentBasedRecommender:
        def __init__(self, movies_df): self.movies_df = movies_df
        def fit(self):
            self.movies_df['genres'] = self.movies_df['genres'].fillna('')
            tfidf = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = tfidf.fit_transform(self.movies_df['genres'])
            self.movie_indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()
        def recommend(self, title, n=10):
            if title not in self.movie_indices: return []
            idx = self.movie_indices[title]
            sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
            sim_indices = sim_scores.argsort()[-n-1:-1][::-1]
            return self.movies_df['title'].iloc[sim_indices].tolist()

class Evaluator:
    def __init__(self, movie_data_path, rating_data_path):
        """Initializes the evaluator and the recommender it will evaluate."""
        self.recommender = HybridRecommender(movie_data_path, rating_data_path)
        self.movies_df = self.recommender.movies_df
        self.ratings_df = self.recommender.ratings_df
        self.popularity_scores = None
        self.title_to_id = None

    def fit_recommender(self, train_df):
        """Fits the recommender and pre-calculates necessary data for evaluation."""
        print("Fitting the recommender model...")
        self.train_df = train_df
        self.recommender.fit(train_df) # SOLVED tränar på ALLA ratings, exkluderar seen movies
        print("Recommender fitted.")
        
        print("Pre-calculating popularity scores...")
        # Calculate popularity as the fraction of users who have rated each movie.
        rating_counts = self.train_df['movieId'].value_counts()
        num_users = self.train_df['userId'].nunique()
        self.popularity_scores = rating_counts / num_users

        # Ensure we only get a single movieId
        movies_unique_titles = self.movies_df.drop_duplicates(subset='title')
        self.title_to_id = pd.Series(movies_unique_titles.movieId.values, index=movies_unique_titles.title)
        print("Popularity scores calculated.")

    def generate_all_recommendations(self):
        """Generates recommendations for all users to create a dataset for evaluation."""
        print("Generating recommendations for all users... (This may take a moment)")
        all_recommendations = {}
        # - life is like a box of chocolates...
        movie_seed = "Forrest Gump (1994)" # Ta en populär film
        #movie_seed = "Nausicaä of the Valley of the Wind (Kaze no tani no Naushika)"

        for user_id in self.train_df['userId'].unique():
            recs = self.recommender.recommend(user_id, movie_seed, 10)
            all_recommendations[user_id] = recs
        print("All recommendations generated.")
        return all_recommendations

    def calculate_precision_at_k(self, all_recommendations, test_df, k=10):
        """
        Calculates the average Precision@k for all users.
        Mäter hur stor andel av de rekommenderade artiklarna som faktiskt är relevanta för användaren.
        Hur stor del är 'korrekta'. Noggrannheten.
        """

        ''' MISTAKE SOLUTION
        Precision mäter om rekommendationerna finns i användarens "relevant items".
        Systemet rekommenderar INTE filmer som användaren redan sett.
        Om all data används till träning kommer systemet alltså exkludera alla sedda filmer.
        Systemet kollar sen att kolla om dess rekommendationer matchar någon sedd film, som inte finns kvar.
        TRAIN TEST!
        Train - filmer systemet vet om och exkluderar från sina rekommendationer.
        Test - filmer systemet INTE vet om och KAN rekommendera.
        '''

        user_precision_points = []

        for user_id, recommendations in all_recommendations.items():
            # Använd INTE hela datasetet
            _seen_movies = test_df.loc[ test_df['userId'] == user_id ]
            _liked_movies = _seen_movies.loc[_seen_movies['rating'] > 2.5 ]
            _liked_movie_ids = set(_liked_movies['movieId'])

            _hits = 0 # antalet rekommendationer som användaren redan gillat
            for rec in recommendations:
                # hämtar ID:n på rekommendationens titel
                movie_id = int( self.title_to_id.get(str(rec)))

                if movie_id is None:
                    print(f"WARNING, {self.title_to_id.get( str(rec).strip() )} not found in titleToId ")
                    continue

                # kollar om ID:n finns i vårt set
                # all data till träning -> blir alltid FALSE eftersom seen_movies exkluderas från rekommendationerna (right?)
                if movie_id in _liked_movie_ids:
                    _hits += 1
                    #print(f"HITS {_hits}")

            _precision = _hits / k
            user_precision_points.append(_precision)

        # calculate average precision
        print(f"...calculating avg precision at k")
        avg_precision_at_k = sum(user_precision_points) / len(user_precision_points)
        print(avg_precision_at_k)

        if user_precision_points:
            return avg_precision_at_k
        else:
            print("---Insufficient data to calculate Precision@k.---")
            return 0.0


    def calculate_coverage(self, all_recommendations):
        """
        Calculates the catalog coverage of the recommendations.
        Hur stor andel av det totala antalet objekt som rekommenderas.
        """
        # varje film som blivit rekommenderad minst 1 gång
        recommended_titles = set()
        
        for _recommandations in all_recommendations.values():
            recommended_titles.update(_recommandations)

        # alla filmer som finns i vår data
        all_titles = set(self.movies_df['title'].unique())

        # andelen av alla filmer som rekommenderats
        print("...calculating coverage")
        coverage = len(recommended_titles) / len(all_titles)
        print(f"{coverage:.4f}, {100*coverage:.2f}%")        
        
        if coverage:
            return coverage
        else:
            return 0.0
    

    def calculate_novelty(self, all_recommendations):
        """
        Calculates the average novelty of the recommendations.
        Mäter hur överraskande eller oväntade rekommendationerna är.
        """
        # varje användares avg. nyhetspoäng
        user_novelty_scores = []

        for user_id, recommendations in all_recommendations.items():
            _novelty_scores = []

            for movie_title in recommendations:
                _movie_id = self.title_to_id.get(movie_title)
                _popularity = self.popularity_scores.get(_movie_id, 0)

                if _popularity > 0:
                    _novelty = -math.log2(_popularity)
                    _novelty_scores.append(_novelty)
            
            # Användarens avg. novelty
            if _novelty_scores:
                _user_avg_novelty = sum(_novelty_scores) / len(_novelty_scores)
                user_novelty_scores.append(_user_avg_novelty)

        # alla användares avg. novelty
        print("...calculating novelty")
        avg_novelty = sum(user_novelty_scores) / len(user_novelty_scores)
        print(avg_novelty)

        if user_novelty_scores:
            return avg_novelty
        else:
            return 0.0

if __name__ == '__main__':
    evaluator = Evaluator(
        movie_data_path='data/movies.csv',
        rating_data_path='data/ratings.csv'
    )

    # dela ratings_df i tränings- och testdata, från "self".rating_df, för varje användare
    train_data = []
    test_data = []

    for user_id in evaluator.ratings_df['userId'].unique():
        user_ratings = evaluator.ratings_df[ evaluator.ratings_df['userId'] == user_id ]
        train, test = train_test_split(user_ratings, test_size=0.2, random_state=42)
        train_data.append(train)
        test_data.append(test)

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    # träna med träningsdata
    evaluator.fit_recommender(train_df)

    all_recs = evaluator.generate_all_recommendations()

    precision = evaluator.calculate_precision_at_k(all_recs, test_df)
    coverage = evaluator.calculate_coverage(all_recs)
    novelty = evaluator.calculate_novelty(all_recs)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Average Precision@10: {precision:.4f}, {100*precision:.2f}%")
    print(f"Catalog Coverage: {coverage:.4f}, {100*coverage:.2f}%")
    print(f"Average Novelty: {novelty:.4f}")