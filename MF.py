import json
import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise import Reader
import pandas as pd
from surprise.model_selection import cross_validate

root_folder = "Data/"

def getInfo(filename):#Assuming each line is a dictonary
    results = []
    with open(root_folder+filename) as file:
        for line in file:
            dict = json.loads(line)
            results.append(dict)
    return results

movieInfo = getInfo("movies.json")
ratingsInfo = getInfo("ratings.json")

def getData():
    raw_data = {'user_id':[], 'movie_id':[], 'rating':[]}
    for a_user in ratingsInfo:
        user_id = a_user['_id']
        ratedMovieDict = a_user['rated']
        for movie_id, rating in ratedMovieDict.items():
            if movie_id == "submit":
                continue
            raw_data['user_id'].append(user_id)
            raw_data['movie_id'].append(movie_id)
            raw_data['rating'].append(int(rating[0])+2) #3 --> 1, 2 --> 0, 1 --> -1

    df = pd.DataFrame(raw_data)
    reader = Reader(rating_scale=(1, 3))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    return data

def main():
    data = getData()

    # We'll use the famous SVD algorithm.
    algo = SVD()

    # Run 5-fold cross-validation and print results
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

main()