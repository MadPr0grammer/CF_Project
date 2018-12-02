import createClassificationFiles

rootFolder = "FlickscoreData-26oct2018/"
testInputFiles = ["fold1_test.json", "fold2_test.json", "fold3_test.json", "fold4_test.json", "fold5_test.json"]
trainInputFiles = ["fold1_train.json", "fold2_train.json", "fold3_train.json", "fold4_train.json", "fold5_train.json"]
testLabelFiles = ["testLabel_1.csv", "testLabel_2.csv", "testLabel_3.csv", "testLabel_4.csv", "testLabel_5.csv"]
trainLabelFiles = ["trainLabel_1.csv", "trainLabel_2.csv", "trainLabel_3.csv", "trainLabel_4.csv", "trainLabel_5.csv"]
trainClassifierFiles = ["train_1.csv", "train_2.csv", "train_3.csv", "train_4.csv", "train_5.csv"]
testClassifierFiles = ["test_1.csv", "test_2.csv", "test_3.csv", "test_4.csv", "test_5.csv"]

for i, filename in enumerate(trainInputFiles):
    createClassificationFiles.createClassifierFiles(filename, trainClassifierFiles[i], trainLabelFiles[i])

for i, filename in enumerate(testInputFiles):
    createClassificationFiles.createClassifierFiles(filename, testClassifierFiles[i], testLabelFiles[i])