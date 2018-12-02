import sol
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

def main():
    Y_original = []
    Y_predicted = []
    test_data = sol.getTestData('test.json')
    trained_models = sol.getTrainedModels(sol.getTrainingSets(sol.getClusters())) # trained on 'ratings.json'
    for a_dict in test_data:
        user_id = a_dict['_id']
        ratedMovieDict = a_dict['rated']
        for movie_id, rating in ratedMovieDict.items():
            if movie_id == "submit":
                continue
            pred = sol.getPrediction(user_id, movie_id, trained_models)
            if math.isnan(pred):
                continue
            Y_predicted.append(pred)
            Y_original.append(int(rating[0]) + 2)

    mae = mean_absolute_error(Y_original, Y_predicted)
    rmse = math.sqrt(mean_squared_error(Y_original, Y_predicted))

    # print(Y_original)
    # print("")
    # print(Y_predicted)
    # print("===============================================")
    # print("")
    print("MAE:", mae)
    print("")
    print("RMSE:", rmse)


main()