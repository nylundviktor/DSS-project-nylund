import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

class Recommender:
    def __init__(self, games_data_path, recommendations_data_path):
        self.games_df = pd.read_csv(games_data_path)
        self.recommendations_df = pd.read_csv(recommendations_data_path)
        self.content_recommender = self._ContentBasedRecommender(self.games_df)
        self.recommendations_df['is_recommended'] = self.recommendations_df['is_recommended'].astype(int)
        # 'features' column from game attributes (rating, positive ratio, platforms)
        # this text is used by the content-based recommender's TF-IDF vectorizer to find similarity
        self.games_df['features'] = (
            self.games_df['rating'].fillna('') + ' ' +
            self.games_df['positive_ratio'].astype(str) + ' ' +
            self.games_df['win'].apply(lambda x: 'win' if x else '') + ' ' +
            self.games_df['mac'].apply(lambda x: 'mac' if x else '') + ' ' +
            self.games_df['linux'].apply(lambda x: 'linux' if x else '')
        )

        self.nmf_model = None; self.user_item_matrix = None
        self.user_mapper = None; self.game_mapper = None
        self.game_inv_mapper = None; self.recs_train_df = None

    def fit(self, recs_train_df):
            self.recs_train_df = recs_train_df # recommendations, user-game interaction
            self.content_recommender.fit()
            self.user_item_matrix = recs_train_df.pivot_table(index='user_id', columns='app_id', values='is_recommended').fillna(0)
            self.user_mapper = {uid: i for i, uid in enumerate(self.user_item_matrix.index)}
            self.game_mapper = {mid: i for i, mid in enumerate(self.user_item_matrix.columns)}
            self.nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=500)
            self.nmf_model.fit(self.user_item_matrix)

    def recommend(self, user_id, game_title_seed, num_recommendations=10):
        content_recs = self.content_recommender.recommend(game_title_seed, num_recommendations)
        collaborative_recs = []
        if user_id in self.user_mapper:
            user_idx = self.user_mapper[user_id]
            user_vector = self.user_item_matrix.loc[user_idx].values.reshape(1, -1)
            user_P = self.nmf_model.transform(user_vector)
            item_Q = self.nmf_model.components_
            predicted_scores = np.dot(user_P, item_Q).flatten()
            scores_series = pd.Series(predicted_scores, index=self.user_item_matrix.columns)
            rated_games = self.recommendations_df[self.recommendations_df['user_id'] == user_id]['app_id']
            scores_series = scores_series.drop(index=rated_games, errors='ignore')
            top_game_ids = scores_series.nlargest(num_recommendations).index.tolist()
            collaborative_recs = self.games_df[self.games_df['app_id'].isin(top_game_ids)]['title'].tolist()
        combined_recs = collaborative_recs + content_recs
        unique_recs = list(dict.fromkeys(combined_recs).keys())
        return unique_recs[:num_recommendations]


    class _ContentBasedRecommender:
        def __init__(self, games_df): 
            self.games_df = games_df
            self.tfidf_matrix = None
            self.game_indices = None
        
        def fit(self):
            self.games_df['features'] = self.games_df['features'].fillna('')
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.games_df['features'])
            self.game_indices = pd.Series(self.games_df.index, index=self.games_df['title']).drop_duplicates()
        
        def recommend(self, game_title, num_recommendations=10):
            if game_title not in self.game_indices: 
                return []
            game_index = self.game_indices[game_title]
            game_vector = self.tfidf_matrix[game_index]
            similarity_scores = cosine_similarity(game_vector, self.tfidf_matrix).flatten()
            
            game_similarity_list = list(enumerate(similarity_scores))
            sorted_similar_games = sorted(game_similarity_list, key=lambda x: x[1], reverse=True)
            
            top_game_indices = [i[0] for i in sorted_similar_games[1:num_recommendations + 1]]
            return self.games_df['title'].iloc[top_game_indices].tolist()


if __name__ == '__main__':
    # Create an instance of the recommender
    recommender = Recommender(
        games_data_path='./data/games.csv', 
        recommendations_data_path='./data/recommendations.csv'
    )

    # Count number of ratings per user
    user_counts = recommender.recommendations_df['user_id'].value_counts()

    # Filter users with at least 3 ratings
    active_users = user_counts[user_counts >= 3].index

    # Filter recommendations to only include those users
    filtered_recommendations = recommender.recommendations_df[
        recommender.recommendations_df['user_id'].isin(active_users)
    ]
    recommender.recommendations_df = filtered_recommendations

    # REMINDER fit with training data = recommendations
    train_data = []
    test_data = []

    # group() is faster
    for user_id, user_ratings in recommender.recommendations_df.groupby('user_id'):
        if len(user_ratings) < 2:
            # Cannot split, only one row or ratings
            test_data.append(user_ratings)
            continue
        train, test = train_test_split(user_ratings, test_size=0.2, random_state=42)
        train_data.append(train)
        test_data.append(test)

    print(recommender.recommendations_df['user_id'].nunique())

    '''
    for user_id in recommender.recommendations_df['user_id'].unique()[:50]:
        user_ratings = recommender.recommendations_df[ recommender.recommendations_df['user_id'] == user_id ]
        if len(user_ratings) < 2:
            # Cannot split, only one row or ratings
            test_data.append(user_ratings)
            continue
        train, test = train_test_split(user_ratings, test_size=0.2, random_state=42)
        train_data.append(train)
        test_data.append(test)
    '''

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    recommender.fit(train_df)
    
    test_user_id=2586
    test_game_title="LIMBO"

    recommendations = recommender.recommend(test_user_id, test_game_title)
    
    if recommendations:
        print(f"\nRecommendations for user {test_user_id} based on '{test_game_title}':")
        for _recommendation in recommendations:
            print(f"- {_recommendation}")
    else:
        print("No recommendations found.")
