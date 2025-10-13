import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

class Recommender:
    def __init__(self, movie_data_path, rating_data_path):
        self.movies_df = pd.read_csv(movie_data_path)
        self.ratings_df = pd.read_csv(rating_data_path)
        self.content_recommender = self._ContentBasedRecommender(self.movies_df)
        
        self.nmf_model = None; self.user_item_matrix = None; self.user_mapper = None
        self.movie_mapper = None; self.movie_inv_mapper = None

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
        def __init__(self, movies_df): 
            self.movies_df = movies_df
            self.tfidf_matrix = None
            self.movie_indices = None
        
        def fit(self):
            self.movies_df['genres'] = self.movies_df['genres'].fillna('')
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.movies_df['genres'])
            self.movie_indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()
        
        def recommend(self, movie_title, num_recommendations=10):
            if movie_title not in self.movie_indices: return []
            movie_index = self.movie_indices[movie_title]
            movie_vector = self.tfidf_matrix[movie_index]
            similarity_scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
            movie_similarity_list = list(enumerate(similarity_scores))
            sorted_similar_movies = sorted(movie_similarity_list, key=lambda x: x[1], reverse=True)
            top_movie_indices = [i[0] for i in sorted_similar_movies[1:num_recommendations + 1]]
            return self.movies_df['title'].iloc[top_movie_indices].tolist()


if __name__ == '__main__':
    print("---Collaborative filtering recommendations---\n")
    # Simple main function to test
    # Create an instance of the recommender
    recommender = Recommender(movie_data_path='./data/movies.csv', rating_data_path='./data/ratings.csv')
    
    # Fit the model
    recommender.fit()
    
    # Get and print recommendations for a movie and user
    #movie_title_to_test = "Nausicaä of the Valley of the Wind (Kaze no tani no Naushika) (1984)"
    movie_title_to_test = "Toy Story (1995)"
    user_id_to_test = 1

    recommendations = recommender.recommend(user_id_to_test, movie_title_to_test)
    
    if recommendations:
        print(f"\nRecommendations for user {user_id_to_test} based on '{movie_title_to_test}':")
        for movie in recommendations:
            print(f"- {movie}")
    else:
        print("No recommendations found.")
