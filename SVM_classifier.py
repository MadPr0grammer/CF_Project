from sklearn import svm
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
for i, filename in enumerate(trainClassifierFiles):
    X = np.loadtxt(filename, delimiter=',')
    Y = np.loadtxt(trainLabelFiles[i], delimiter=',')
    model = svm.SVC(kernel='rbf', gamma=0.6, C=1.0)
    svm_clf = model.fit(X, Y )
    X_test = np.loadtxt(testClassifierFiles[i], delimiter=',')
    Y_predicted = svm_clf.predict(X_test)
    Y_original = np.loadtxt(testLabelFiles[i], delimiter=',')
    nm = mean_absolute_error(Y_predicted, Y_original) / 2
    nmae.append(nm)
    rm = math.sqrt(mean_squared_error(Y_predicted, Y_original))
    rmse.append(rm)

print("NMAE:")
print(nmae)
print("Average NMAE: "+str(np.mean(nmae)))
print("")
print("RMSE")
print(rmse)
print("Average RMSE: "+str(np.mean(rmse)))



