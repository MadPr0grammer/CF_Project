import json
import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise import Reader
import pandas as pd
import math

root_folder = "Data/"


def getTestData(filename): #return  a list of dict
    with open(root_folder + filename) as file:
        testData = json.load(file)
        return testData


#for getting user and movie info
def getInfo(filename):#Assuming each line is a dictonary
    results = []
    with open(root_folder+filename) as file:
        for line in file:
            dict = json.loads(line)
            results.append(dict)
    return results

#usersInfo = getInfo("users.json")
movieInfo = getInfo("movies.json")
ratingsInfo = getInfo("ratings.json")

def getMovieGenre(movie_id):
    for movie in movieInfo:
        if (movie['movie_id'] == movie_id):
            genres = movie['genre']
            return genres

def getGenreIndex(genre):
    if genre == 'Music':
        genre = 'Musical'
    genres = ['Biography', 'Horror', 'Crime', 'Romance', 'Action', 'Animation', 'Musical', 'Adventure', 'News', 'Western', 'Comedy', 'War', 'Thriller', 'Sci-Fi', 'Mystery', 'History', 'Drama', 'Sport', 'Fantasy', 'Family']
    return genres.index(genre)

# we already know that there are 20 genres, therefore there will be 20 clusters
# Biography -> 0, Horror -> 1, Crime -> 2, Romance -> 3, Action -> 4, Animation -> 5, Musical -> 6, Adventure -> 7, News -> 8, Western -> 9, Comedy -> 10, War -> 11, Thriller -> 12, Sci-Fi -> 13, Mystery -> 14, History -> 15, Drama -> 16, Sport -> 17, Fantasy -> 18, Family -> 19
def getClusters():
    cluster_list = [{'user_id':[], 'movie_id':[], 'rating':[]} for x in range(20)] # dict at index i represents i-th cluster corresponding to the ith genre
    for a_user in ratingsInfo:
        user_id = a_user['_id']
        ratedMovieDict = a_user['rated']
        for movie_id, rating in ratedMovieDict.items():
            if movie_id == "submit":
                continue
            genres = getMovieGenre(movie_id)
            for genre in genres:
                cluster_index = getGenreIndex(genre)
                cluster_list[cluster_index]['user_id'].append(user_id)
                cluster_list[cluster_index]['movie_id'].append(movie_id)
                cluster_list[cluster_index]['rating'].append(int(rating[0])+2) #3 --> 1, 2 --> 0, 1 --> -1

    return cluster_list

def getTrainingSets(clusters):#traning set is different from dataset, clusters is a list of dict for each cluster
    trainsets = []
    for a_cluster in clusters:
        df = pd.DataFrame(a_cluster)
        reader = Reader(rating_scale=(1, 3))
        data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        trainsets.append(trainset)
    return trainsets

def getTrainedModels(trainsets): #trainsets is a list of tainingset
    models = []
    for a_trainset in trainsets:
        model = SVD()
        model = model.fit(a_trainset)
        models.append(model)
    return models

def getPrediction(user_id, movie_id, trained_models): # Here trained_models is a list of models, each corresponding to  a cluster
    genres = getMovieGenre(movie_id)
    predictions = []
    for genre in genres:
        cluster_index = getGenreIndex(genre)
        model = trained_models[cluster_index]
        prediction = model.predict(user_id, movie_id)
        print(prediction[3])
        predictions.append(prediction[3])
    return np.mean(predictions)


