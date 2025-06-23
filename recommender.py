import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD

# Load and process data
recipes = pd.read_csv('RAW_recipes.csv')
interactions = pd.read_csv('RAW_interactions.csv')

recipes = recipes[recipes['ingredients'].apply(lambda x: isinstance(x, str))]
recipes['ingredients'] = recipes['ingredients'].apply(lambda x: ' '.join(ast.literal_eval(x.lower())))
breakfast_recipes = recipes[recipes['tags'].str.contains('breakfast', case=False, na=False)].copy()

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(breakfast_recipes['ingredients'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Content-based
def recommend_recipe(title):
    if title not in breakfast_recipes['name'].values:
        return ["Recipe not found."]
    idx = breakfast_recipes[breakfast_recipes['name'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    indices = [i[0] for i in sim_scores]
    return breakfast_recipes['name'].iloc[indices].tolist()

# Collaborative filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions[['user_id', 'recipe_id', 'rating']], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

def recommend_for_user(user_id, n=5):
    recipe_ids = breakfast_recipes['id'].tolist()
    predictions = [(recipe_id, model.predict(user_id, recipe_id).est) for recipe_id in recipe_ids]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    top_ids = [i[0] for i in top_n]
    return breakfast_recipes[breakfast_recipes['id'].isin(top_ids)]['name'].tolist()

# Expose data for UI dropdowns
def get_recipe_names():
    return breakfast_recipes['name'].dropna().unique().tolist()

def get_user_ids():
    return interactions['user_id'].unique().tolist()
