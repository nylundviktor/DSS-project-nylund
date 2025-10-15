import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

def reviews_to_category(n):
    if n < 100:
        return 'few_reviews'
    elif n < 1000:
        return 'some_reviews'
    elif n < 10000:
        return 'many_reviews'
    else:
        return 'popular'

def price_to_category(price):
    if price == 0:
        return 'free'
    elif price < 5:
        return 'cheap'
    elif price < 20:
        return 'midrange'
    else:
        return 'expensive'


class Recommender:
    def __init__(self, games_data_path, recommendations_data_path):
        self.games_df = pd.read_csv(games_data_path)
        self.recommendations_df = pd.read_csv(recommendations_data_path)

        self.content_recommender = self.ContentBasedRecommender(self.games_df)
        self.collaborative_recommender = self.CollaborativeRecommender(self.games_df, self.recommendations_df)
    
        self.recommendations_df['is_recommended'] = self.recommendations_df['is_recommended'].astype(int)
        # 'features' column from game attributes (rating, positive ratio, platforms)
        # this text is used by the content-based recommender's TF-IDF vectorizer to find similarity
        self.games_df['features'] = (
            self.games_df['rating'].fillna('') + ' ' +
            self.games_df['positive_ratio'].astype(str) + ' ' +
            self.games_df['price_final'].apply(price_to_category) + ' ' +
            self.games_df['user_reviews'].apply(reviews_to_category) + ' ' +
            self.games_df['win'].apply(lambda x: 'win' if x else '') + ' ' +
            self.games_df['mac'].apply(lambda x: 'mac' if x else '') + ' ' +
            self.games_df['linux'].apply(lambda x: 'linux' if x else '')
        )

    def fit(self, recs_train_df):
            #self.recs_train_df = recs_train_df # recommendations, user-game interaction
            self.content_recommender.fit()
            self.collaborative_recommender.fit(recs_train_df)

    def recommend(self, user_id, game_title_seed, num_recommendations=10):
        content_recs = self.content_recommender.recommend(game_title_seed, num_recommendations)
        collaborative_recs = self.collaborative_recommender.recommend(user_id, num_recommendations) 

        combined_recs = collaborative_recs + content_recs
        #unique_recs = list(dict.fromkeys(combined_recs).keys())
        unique_recs = [r for r in dict.fromkeys(combined_recs).keys() if r != game_title_seed]
        return unique_recs[:num_recommendations]
    
    
    # Collaborative filtering (user interaction data, finds games from patterns between users)
    class CollaborativeRecommender:
        def __init__(self, games_df, recommendations_df):
            self.games_df = games_df
            self.recommendations_df = recommendations_df

            self.nmf_model = None; self.user_item_matrix = None
            self.user_mapper = None; self.game_mapper = None
            #self.game_inv_mapper = None

        def fit(self, recs_train_df):
            self.recs_train_df = recs_train_df # recommendations, user-game interaction
            #self.user_item_matrix = recs_train_df.pivot_table(index='user_id', columns='app_id', values='is_recommended').fillna(0)
            pivot = recs_train_df.pivot_table(
                index='user_id', columns='app_id', values='is_recommended', fill_value=0
            )
            pivot = pivot.astype(pd.SparseDtype("float", fill_value=0)) # Convert to a sparse DataFrame
            pivot_sparse = pivot.sparse.to_coo()
            self.user_item_matrix = pivot


            self.user_mapper = {uid: i for i, uid in enumerate(pivot.index)}
            self.game_mapper = {mid: i for i, mid in enumerate(pivot.columns)}

            self.nmf_model = NMF(n_components=20, init='random', solver='mu', random_state=42, max_iter=200)
            self.nmf_model.fit(pivot_sparse)
            #self.user_mapper = {uid: i for i, uid in enumerate(self.user_item_matrix.index)}
            #self.game_mapper = {mid: i for i, mid in enumerate(self.user_item_matrix.columns)}
            #self.nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=500)
            #self.nmf_model.fit(self.user_item_matrix)

        def recommend(self, user_id, num_recommendations=10):
            collaborative_recs = []
            if user_id in self.user_mapper:
                #user_idx = self.user_mapper[user_id]
                #user_vector = self.user_item_matrix.loc[user_idx].values.reshape(1, -1)
                user_vector = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
                user_P = self.nmf_model.transform(user_vector)
                item_Q = self.nmf_model.components_
                predicted_scores = np.dot(user_P, item_Q).flatten()
                scores_series = pd.Series(predicted_scores, index=self.user_item_matrix.columns)
                rated_games = self.recommendations_df[self.recommendations_df['user_id'] == user_id]['app_id']
                scores_series = scores_series.drop(index=rated_games, errors='ignore')
                top_game_ids = scores_series.nlargest(num_recommendations).index.tolist()
                collaborative_recs = self.games_df[self.games_df['app_id'].isin(top_game_ids)]['title'].tolist()
            return collaborative_recs


    # Content based filtering (attributes and features, finds similar games)
    class ContentBasedRecommender:
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

    # sample 1, filtered
    user_counts = recommender.recommendations_df['user_id'].value_counts()
    print(f"User left: {len(user_counts)}")
    active_users = user_counts[user_counts >= 10].index
    filtered_sample = active_users.to_series().sample(n=60000, random_state=42)
    filtered_recommendations = recommender.recommendations_df[
        recommender.recommendations_df['user_id'].isin(filtered_sample)
    ]
    print(f"After filtering: {len(filtered_recommendations)}")
    
    # sample 2, random
    all_users = user_counts.index
    random_users = all_users.to_series().sample(n=10000, random_state=42)
    random_sample = recommender.recommendations_df[
        recommender.recommendations_df['user_id'].isin(random_users)
    ]
    # Combine samples
    combined_sample = pd.concat([filtered_recommendations, random_sample]).drop_duplicates()
    recommender.recommendations_df = combined_sample
    print(f"Combined sample size (rows): {len(combined_sample)}")
    print(f"Unique users in combined sample: {combined_sample['user_id'].nunique()}")
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

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    recommender.fit(train_df)

    test_user_id=2586
    test_game_title="LIMBO"

    print(f"\n---Content-Based Recommendations for user {test_user_id} based on '{test_game_title}'---")
    print(recommender.content_recommender(test_game_title))

    print(f"\n---Collaborative Recommendations for user {test_user_id} based on '{test_game_title}'---")
    print(recommender.collaborative_recommender.recommend(test_user_id))

    recommendations = recommender.recommend(test_user_id, test_game_title)

    if recommendations:
        print(f"\n---Recommendations for user {test_user_id} based on '{test_game_title}'---")
        for _recommendation in recommendations:
            print(f"- {_recommendation}")
    else:
        print("No recommendations found.")
