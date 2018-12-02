from hpelm import ELM
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import math

rootFolder = "FlickscoreData-26oct2018/"
testInputFiles = ["fold1_test.json", "fold2_test.json", "fold3_test.json", "fold4_test.json", "fold5_test.json"]
trainInputFiles = ["fold1_train.json", "fold2_train.json", "fold3_train.json", "fold4_train.json", "fold5_train.json"]
testLabelFiles = ["testLabel_1.csv", "testLabel_2.csv", "testLabel_3.csv", "testLabel_4.csv", "testLabel_5.csv"]
trainLabelFiles = ["trainLabel_1.csv", "trainLabel_2.csv", "trainLabel_3.csv", "trainLabel_4.csv", "trainLabel_5.csv"]
trainClassifierFiles = ["train_1.csv", "train_2.csv", "train_3.csv", "train_4.csv", "train_5.csv"]
testClassifierFiles = ["test_1.csv", "test_2.csv", "test_3.csv", "test_4.csv", "test_5.csv"]

nmae = []
rmse = []

def onehotEncode(Y):#assuming 3 classes - <-1,0,1>, Y is a matrix
    onehot = np.zeros((Y.shape[0],3))
    for i, label in enumerate(Y):
        if i[0] == -1:
            onehot[i,0] = 1
        elif i[0] == 0:
            onehot[i,1] = 1
        else:
            onehot[i,2] = 1
    return onehot

def make_predicted_classes(predicted_matrix):
    predicted_classes_matrix = np.zeros((predicted_matrix.shape))
    for i, row in enumerate(predicted_matrix):
        idx_of_max = np.argmax(row)
        predicted_classes_matrix[i,idx_of_max] = 1
    return predicted_classes_matrix

def getLabelsFromOnehot(Y_predicted_onehot):
    result = np.zeros((Y_predicted_onehot.shape[0],1))
    for i, row in enumerate(Y_predicted_onehot):
        for j, col in enumerate(row):
            if(j == 0 and col == 1):
                result[i,0] = -1
                break
            elif(j==1 and col == 1):
                result[i,0] = 0
                break
            elif(j==2 and col == 1):
                result[i,0] = 1
                break
    return result

for i, filename in enumerate(trainClassifierFiles):
    X = np.loadtxt(filename, delimiter=',')
    Y = np.loadtxt(trainLabelFiles[i], delimiter=',')
    Y_onehot = onehotEncode(Y)
    elm = ELM(X.shape[1], Y_onehot.shape[1])
    elm.add_neurons(200, "sigm")
    elm.add_neurons(100, "tanh")
    elm.train(X, Y, "CV", "OP", "c", k=5)
    X_test = np.loadtxt(testClassifierFiles[i], delimiter=',')
    Y_predicted_elm = elm.predict(X_test)
    Y_predicted_onehot = make_predicted_classes(Y_predicted_elm)
    Y_predicted_labels = getLabelsFromOnehot(Y_predicted_onehot)
    Y_original = np.loadtxt(testLabelFiles[i], delimiter=',')
    nm = mean_absolute_error(Y_predicted_labels, Y_original) / 2
    nmae.append(nm)
    rm = math.sqrt(mean_squared_error(Y_predicted_labels, Y_original))
    rmse.append(rm)

print("NMAE:")
print(nmae)
print("Average NMAE: "+str(np.mean(nmae)))
print("")
print("RMSE")
print(rmse)
print("Average RMSE: "+str(np.mean(rmse)))



