import pandas as pd

# Load the MovieLens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Remove duplicates and missing values
movies.drop_duplicates(inplace=True)
ratings.dropna(inplace=True)

# Create a user-item matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')


import seaborn as sns

# Analyze the distribution of ratings and popularity of movies
sns.histplot(ratings['rating'])
sns.histplot(ratings.groupby('movieId')['rating'].mean())

# Explore the correlation between different genres and ratings
movie_genre = movies['genres'].str.get_dummies(sep='|')
movie_genre_ratings = pd.concat([movie_genre, ratings['rating']], axis=1)
sns.heatmap(movie_genre_ratings.corr())

# Identify any trends or patterns in the data
sns.lineplot(data=ratings.groupby('timestamp').mean())

from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD

# Load the data into the Surprise format
reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Implement the SVD algorithm
algo = SVD()
algo.fit(trainset)

# Tune the hyperparameters of the algorithm
from surprise.model_selection import GridSearchCV

param_grid = {'n_epochs': [5, 10, 15], 'lr_all': [0.002, 0.005, 0.01],
              'reg_all': [0.4, 0.6, 0.8]}
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
grid_search.fit(data)

# Get the best hyperparameters and retrain the model
best_params = grid_search.best_params['rmse']
algo = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'],
           reg_all=best_params['reg_all'])
algo.fit(trainset)

from surprise import accuracy

# Evaluate the performance of the model
predictions = algo.test(testset)
print('MAE:', accuracy.mae(predictions))
print('RMSE:', accuracy.rmse(predictions))

# Compare the performance of the model with other algorithms
from surprise import KNNBasic, CoClustering

knn = KNNBasic()
co_cluster = CoClustering()
for model in [algo, knn, co_cluster]:
    print(model.__class__.__name__)
    model.fit(trainset)
    predictions = model.test(testset)
    print('MAE:', accuracy.mae(predictions))
    print('RMSE:', accuracy.rmse(predictions))
    print()


import pickle

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(algo, f)

# Load the model and make recommendations
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make recommendations for a user
user_id = 1
movie_ids = ratings[ratings['userId'] == user_id]['movieId']
unseen_movies = set(user_item_matrix.columns) - set(movie_ids)
unseen_movies = list(unseen_movies)
testset = [[user_id, movie_id, 4.] for movie_id in unseen_movies]
predictions = model.test(testset)
top_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]

# Print the top recommended movies for the user
for i, recommendation in enumerate(top_recommendations):
    movie = movies.loc[movies['movieId'] == recommendation.iid]['title'].values[0]
    print(f"{i + 1}. {movie}")
