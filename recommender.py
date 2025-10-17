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
            
            top_game_indices = [i[0] for i in sorted_similar_games]
            recommended_titles = self.games_df['title'].iloc[top_game_indices].tolist()
            recommended_titles = [title for title in recommended_titles if title != game_title]

            return recommended_titles[:num_recommendations]
        
    
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

            # init='random' / init='nndsvda'
            self.nmf_model = NMF(
                n_components=20, 
                init='nndsvda', 
                solver='mu', 
                random_state=42, 
                max_iter=500)
            self.nmf_model.fit(pivot_sparse)
            #self.user_mapper = {uid: i for i, uid in enumerate(self.user_item_matrix.index)}
            #self.game_mapper = {mid: i for i, mid in enumerate(self.user_item_matrix.columns)}
            #self.nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=500)
            #self.nmf_model.fit(self.user_item_matrix)

        def recommend(self, user_id, num_recommendations=10):
            collaborative_recs = []
            if user_id not in self.user_item_matrix.index:
                return []
            #user_idx = self.user_mapper[user_id]
            #user_vector = self.user_item_matrix.loc[user_idx].values.reshape(1, -1)
            user_vector = self.user_item_matrix.loc[user_id].to_numpy().reshape(1, -1)
            if np.count_nonzero(user_vector) == 0:
                # This user has no usable interaction data
                return []
            
            user_P = self.nmf_model.transform(user_vector)
            item_Q = self.nmf_model.components_
            predicted_scores = np.dot(user_P, item_Q).flatten()
            scores_series = pd.Series(predicted_scores, index=self.user_item_matrix.columns)
            rated_games = self.recs_train_df[self.recs_train_df['user_id'] == user_id]['app_id']
            scores_series = scores_series.drop(index=rated_games, errors='ignore')
            top_game_ids = scores_series.nlargest(num_recommendations).index.tolist()
            collaborative_recs = self.games_df[self.games_df['app_id'].isin(top_game_ids)]['title'].tolist()

            return collaborative_recs


def weighted_hybrid_recommender(user_id, base_game, num_recs=10, alpha=0.7):
    collab_list = recommender.collaborative_recommender.recommend(user_id, num_recs*2)
    content_list = recommender.content_recommender.recommend(base_game, num_recs*2)
    
    # simple inverse-rank scores
    collab_scores = {game: 1/(i+1) for i, game in enumerate(collab_list)}
    content_scores = {game: 1/(i+1) for i, game in enumerate(content_list)}
    
    combined_scores = {}
    for game in set(collab_scores) | set(content_scores):
        c_score = collab_scores.get(game, 0)
        cb_score = content_scores.get(game, 0)
        combined_scores[game] = alpha * c_score + (1 - alpha) * cb_score
    
    sorted_games = sorted(combined_scores, key=combined_scores.get, reverse=True)
    # filter out seed game and limit results
    filtered = [g for g in sorted_games if g != base_game]
    return filtered[:num_recs]

class Evaluator:
    def __init__(self, recommender):
        """Initializes the evaluator with an existing trained recommender instance."""
        self.recommender = recommender
        self.games_df = recommender.games_df
        self.recommendations_df = recommender.recommendations_df

    def fit_recommender(self, train_df, test_df):
        """Fits the recommender and pre-calculates necessary data for evaluation."""
        print("\n---Start evaluation---")
        self.train_df = train_df
        self.test_df = test_df
        
        print("Pre-calculating popularity scores...")
        # Calculate popularity as the fraction of users who rated each game
        game_counts = self.train_df['app_id'].value_counts()
        num_users = self.train_df['user_id'].nunique()
        self.popularity_scores = game_counts / num_users

        # Ensure we don't get duplicates
        unique_games = self.games_df.drop_duplicates(subset='title')
        self.title_to_id = pd.Series(unique_games.app_id.values, index=unique_games.title)
        print("Popularity scores calculated.")

    def generate_all_recommendations(self, seed_game_title="LIMBO"):
        """Generates recommendations for all users to create a dataset for evaluation."""
        print("Generating recommendations for all users... (This may take a moment)")
        all_recommendations = {}
 
        for user_id in self.train_df['user_id'].unique():
            recs = self.recommender.recommend(user_id, seed_game_title, 10)
            all_recommendations[user_id] = recs

        print("All recommendations generated.")
        return all_recommendations

    def calculate_precision_at_k(self, all_recommendations, test_df, k=10):
        """
        Calculates the average Precision@k for all users.
        Mäter hur stor andel av de rekommenderade artiklarna som faktiskt är relevanta för användaren.
        """
        user_precision_points = []

        for user_id, recommendations in all_recommendations.items():
            _reviewed_games = test_df.loc[ test_df['user_id'] == user_id ]
            _liked_games = _reviewed_games.loc[_reviewed_games['is_recommended'] == 1 ]
            _liked_games_ids = set(_liked_games['app_id'])

            _hits = 0 # antalet rekommendationer som användaren gillat
            for rec in recommendations:
                app_id_str = self.title_to_id.get(str(rec))
                if app_id_str is None:
                    print(f"WARNING, {self.title_to_id.get( str(rec).strip() )} not found in titleToId ")
                    continue
                # kollar om ID:n finns i vårt set
                app_id = int(app_id_str)
                if app_id in _liked_games_ids:
                    _hits += 1

            _precision = _hits / k
            user_precision_points.append(_precision)

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
        # varje spel som blivit rekommenderat minst 1 gång
        recommended_titles = set()
        
        for _recommendations in all_recommendations.values():
            recommended_titles.update(_recommendations)

        # alla spel som finns i vår data
        all_titles = set(self.games_df['title'].unique())

        # andelen av alla spel som rekommenderats
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

            for game_title in recommendations:
                _game_id = self.title_to_id.get(game_title)
                _popularity = self.popularity_scores.get(_game_id, 0)

                if _popularity > 0:
                    _novelty = -math.log2(_popularity)
                    _novelty_scores.append(_novelty)
            
            # Användarens avg. novelty
            if _novelty_scores:
                _user_avg_novelty = sum(_novelty_scores) / len(_novelty_scores)
                user_novelty_scores.append(_user_avg_novelty)

        # alla användares avg. novelty
        print("...calculating novelty")
        if user_novelty_scores:
            avg_novelty = sum(user_novelty_scores) / len(user_novelty_scores)
            print(avg_novelty)
            return avg_novelty
        else:
            return 0.0
        

if __name__ == '__main__':
    # Create an instance of the recommender
    recommender = Recommender(
        games_data_path='./data/games.csv', 
        recommendations_data_path='./data/recommendations.csv'
    )

    test_user_id = 4616950
    test_game_title="LIMBO"

    # sample 1, filtered
    user_counts = recommender.recommendations_df['user_id'].value_counts()
    print(f"User in data: {len(user_counts)}")
    active_users = user_counts[user_counts >= 10].index
    n1 = 30000
    filtered_sample = active_users.to_series().sample(n=n1, random_state=42)
    filtered_sample = pd.concat([filtered_sample, pd.Series([test_user_id])]).drop_duplicates()
    filtered_recommendations = recommender.recommendations_df[
        recommender.recommendations_df['user_id'].isin(filtered_sample)
    ]
    print(f"After filtering: {len(filtered_recommendations)}")
    # sample 2, random
    all_users = user_counts.index
    n2 = 5000
    random_users = all_users.to_series().sample(n=n2, random_state=42)
    random_sample = recommender.recommendations_df[
        recommender.recommendations_df['user_id'].isin(random_users)
    ]
    # Combine samples
    combined_sample = pd.concat([filtered_recommendations, random_sample]).drop_duplicates()
    recommender.recommendations_df = combined_sample
    print(f"Combined sample size: {len(combined_sample)}")
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

    user_interactions = recommender.recommendations_df[recommender.recommendations_df['user_id'] == test_user_id]
    print(f"User {test_user_id} reviews: {len(user_interactions)}")
    if test_user_id in recommender.collaborative_recommender.user_mapper:
        print(f"User {test_user_id} is known to the collaborative model.")
    else:
        print(f"User {test_user_id} NOT found in collaborative model.")


    print(f"\n---Content-Based Recommendations based on '{test_game_title}'---")
    content_recs = recommender.content_recommender.recommend(test_game_title)
    for _rec in content_recs:
        print(f"- {_rec}")

    print(f"\n---Collaborative Recommendations for user {test_user_id}---")
    collaborative_recs = recommender.collaborative_recommender.recommend(test_user_id)
    for _rec in collaborative_recs:
        print(f"- {_rec}")

    print(f"\n---Hybrid Recommendations for user {test_user_id} based on '{test_game_title}'---")
    hybrid_recs = recommender.recommend(test_user_id, test_game_title)
    for _rec in hybrid_recs:
        print(f"- {_rec}")

    '''
    print(f"\n---Weighted Hybrid Recommendations for user {test_user_id} based on '{test_game_title}'---")
    whybrid_recs = weighted_hybrid_recommender(test_user_id, test_game_title)
    for rec in whybrid_recs:
        print(f"- {rec}")
    '''

    evaluator = Evaluator(recommender)
    evaluator.fit_recommender(train_df, test_df)
    all_recs = evaluator.generate_all_recommendations(seed_game_title="LIMBO")

    precision = evaluator.calculate_precision_at_k(all_recs, test_df, k=10)
    coverage = evaluator.calculate_coverage(all_recs)
    novelty = evaluator.calculate_novelty(all_recs)

    print("\n--- Evaluation Metrics ---")
    print(f"Precision@10: {precision:.4f}, {100*precision:.2f}%")
    print(f"Coverage: {coverage:.4f}, {100*coverage:.2f}%")
    print(f"Novelty: {novelty:.4f}")