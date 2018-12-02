import json
from datetime import datetime
from datetime import date
import numpy as np
root_folder = "FlickscoreData-26oct2018/"


#for getting user and movie info
def getInfo(filename):#Assuming each line is a dictonary
    results = []
    with open(root_folder+filename) as file:
        for line in file:
            dict = json.loads(line)
            results.append(dict)
    return results

usersInfo = getInfo("users.json")
movieInfo = getInfo("movies.json")

# userInfo = getInfo("movies.json")
# print(userInfo[0])

def getAge(stringAge):#getting age as of 01 Jan 2017, cuz dataset was published then
    s = stringAge.split("-")
    nowYear = 2017
    #print(s[2])
    try:
        dob_year = int(s[2])
    except ValueError:
        return 29 #averageage of indian population
    age = nowYear - dob_year
    # now_date = date(2017,1,1)
    # print(stringAge)
    # dob = datetime.strptime(stringAge, "%d-%m-%Y")
    # age = now_date.year - dob.year - ((now_date.month, now_date.day) < (dob.month, dob.day)) #taken from https://stackoverflow.com/a/9754466/6360817
    return age


def getUserFeatureVector(user_id):
    # <Hindi, Bengali, Assamese, Tamil, Nepali, Punjabi, Rajasthani, Malayalam, Bhojpuri, Kannada, Haryanvi, Manipuri , Urdu, Marathi, Telugu, Oriya, Gujarati, Konkani, 0-18, 19-24, 25-34, 35-44, 45-49, 50-55, 56-100, Student, Service, Retried, Self-employed, Other, Male, Female>
    userVector = np.zeros((1,32)) #32 dimensional binary feature vector
    for user in usersInfo:
        if(user['_id'] == user_id):
            #fill in language
            languages = user['languages']
            if "Hindi" in languages:
                userVector[0,0] = 1
            if "Bengali" in languages:
                userVector[0,1] = 1
            if "Assamese" in languages:
                userVector[0,2] = 1
            if "Tamil" in languages:
                userVector[0,3] = 1
            if "Nepali" in languages:
                userVector[0,4] = 1
            if "Punjabi" in languages:
                userVector[0,5] = 1
            if "Rajasthani" in languages:
                userVector[0,6] = 1
            if "Malayalam" in languages:
                userVector[0,7] = 1
            if "Bhojpuri" in languages:
                userVector[0,8] = 1
            if "Kannada" in languages:
                userVector[0,9] = 1
            if "Haryanvi" in languages:
                userVector[0,10] = 1
            if "Manipuri" in languages:
                userVector[0,11] = 1
            if "Urdu" in languages:
                userVector[0,12] = 1
            if "Marathi" in languages:
                userVector[0,13] = 1
            if "Telugu" in languages:
                userVector[0,14] = 1
            if "Oriya" in languages:
                userVector[0,15] = 1
            if "Gujarati" in languages:
                userVector[0,16] = 1
            if "Konkani" in languages:
                userVector[0,17] = 1

            #fill in age
            #print(user['dob'])
            age = getAge(user['dob'])
            if 0 <= age <= 18:
                userVector[0,18] = 1
            elif 19 <= age <= 24:
                userVector[0,19] = 1
            elif 25 <= age <= 34:
                userVector[0,20] = 1
            elif 35 <= age <= 44:
                userVector[0,21] = 1
            elif 45 <= age <= 49:
                userVector[0,22] = 1
            elif 50 <= age <= 55:
                userVector[0,23] = 1
            else:
                userVector[0,24] = 1

            #fill in occupation
            occupation = user['job']
            if occupation == "Student":
                userVector[0,25] = 1
            elif occupation == "Service":
                userVector[0,26] = 1
            elif occupation == "Retried":
                userVector[0,27] = 1
            elif occupation == "Self-employed":
                userVector[0,28] = 1
            else:
                userVector[0, 29] = 1

            #fill in ggender
            gender = user['gender']
            if gender == "Male":
                userVector[0,30] = 1
            else:
                userVector[0,31] = 1

            break
    return userVector

def getMovieFeatureVector(movie_id):
    #<Hindi, Bengali, Assamese, Tamil, Nepali, Punjabi, Rajasthani, Malayalam, Bhojpuri, Kannada, Haryanvi, Manipuri , Urdu, Marathi, Telugu, Oriya, Gujarati, Konkani, Biography, Horror, Crime, Romance, Action, Animation, Musical, Adventure, News, Western, Comedy, War, Thriller, Sci-Fi, Mystery, History, Drama, Sport, Fantasy, Family>
    movieVector = np.zeros((1,38))#38 dimensional vector
    for movie in movieInfo:
        if (movie['movie_id'] == movie_id):
            # fill in language
            languages = movie['language']
            if "Hindi" in languages:
                movieVector[0, 0] = 1
            if "Bengali" in languages:
                movieVector[0, 1] = 1
            if "Assamese" in languages:
                movieVector[0, 2] = 1
            if "Tamil" in languages:
                movieVector[0, 3] = 1
            if "Nepali" in languages:
                movieVector[0, 4] = 1
            if "Punjabi" in languages:
                movieVector[0, 5] = 1
            if "Rajasthani" in languages:
                movieVector[0, 6] = 1
            if "Malayalam" in languages:
                movieVector[0, 7] = 1
            if "Bhojpuri" in languages:
                movieVector[0, 8] = 1
            if "Kannada" in languages:
                movieVector[0, 9] = 1
            if "Haryanvi" in languages:
                movieVector[0, 10] = 1
            if "Manipuri" in languages:
                movieVector[0, 11] = 1
            if "Urdu" in languages:
                movieVector[0, 12] = 1
            if "Marathi" in languages:
                movieVector[0, 13] = 1
            if "Telugu" in languages:
                movieVector[0, 14] = 1
            if "Oriya" in languages:
                movieVector[0, 15] = 1
            if "Gujarati" in languages:
                movieVector[0, 16] = 1
            if "Konkani" in languages:
                movieVector[0, 17] = 1

            #fill in genre
            genre = movie['genre']
            if "Biography" in genre:
                movieVector[0, 18] = 1
            if "Horror" in genre:
                movieVector[0, 19] = 1
            if "Crime" in genre:
                movieVector[0, 20] = 1
            if "Romance" in genre:
                movieVector[0, 21] = 1
            if "Action" in genre:
                movieVector[0, 22] = 1
            if "Animation" in genre:
                movieVector[0, 23] = 1
            if "Musical" in genre:
                movieVector[0, 24] = 1
            if "Adventure" in genre:
                movieVector[0, 25] = 1
            if "News" in genre:
                movieVector[0, 26] = 1
            if "Western" in genre:
                movieVector[0, 27] = 1
            if "Comedy" in genre:
                movieVector[0, 28] = 1
            if "War" in genre:
                movieVector[0, 29] = 1
            if "Thriller" in genre:
                movieVector[0, 30] = 1
            if "Sci-Fi" in genre:
                movieVector[0, 31] = 1
            if "Mystery" in genre:
                movieVector[0, 32] = 1
            if "History" in genre:
                movieVector[0, 33] = 1
            if "Drama" in genre:
                movieVector[0, 34] = 1
            if "Sport" in genre:
                movieVector[0, 35] = 1
            if "Fantasy" in genre:
                movieVector[0, 36] = 1
            if "Family" in genre:
                movieVector[0, 37] = 1

            break
    return movieVector


#creates training file and label file for classifier from the fold_train input files
def createClassifierFiles(inputFilename, trainFilename, labelFilename):
    with open(root_folder+inputFilename) as inputFile, open(trainFilename, 'w+') as trainFile, open(labelFilename, 'w+') as labelFile:
        inputData = json.load(inputFile)
        for a_dict in inputData:
            userFeatures = getUserFeatureVector(a_dict['_id'])
            ratedMovieDict = a_dict['rated']
            for movie_id, rating in ratedMovieDict.items():
                if movie_id == "submit":
                    continue
                movieFeatures = getMovieFeatureVector(movie_id)
                a_datapoint_for_classification = np.concatenate([userFeatures[0],movieFeatures[0]])
                trainFile.write(','.join(map(str, a_datapoint_for_classification)))
                trainFile.write("\n")
                labelFile.write(rating[0]+"\n")#cuz rating is stored as list of 1 element in the json file

# with open(root_folder+"fold1_train.json") as trainfile:
#     data = json.load(trainfile)
#     #print(data[0]['rated']['submit'])
#     print(data[0])
