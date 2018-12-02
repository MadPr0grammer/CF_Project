# CF_Project

Contributors:
  Saurabh Kumar 2015089
  Shyam Agrawal 2015099
  
We implemented the paper titled: Weighting strategies for a recommender system using item clustering based on genres
Published on 1 July 2017 in Expert Systems With Applications
Link: https://www.sciencedirect.com/science/article/pii/S0957417417300404?via%3Dihub

We tested the proposed algorithm on Indian Regional Movie Dataset (link: https://arxiv.org/abs/1801.02203)

We also ran 4 different algorithms on this dataset to compare the proposed algorithm with them.
The algorithms ran are:
  1) SVD Matrix Factorization (MF.py)
  2.) Cold-start based classification using SVM (SVM_classifier.py) [1]
  3.) Cold-start based classification using MLP (MLP.py) [1]
  4.) Cold-start based classification using ELM (ELM.py) [1]
  5.) Content-based clustering (on genre) and CF (sol.py and driver.py) [2]




Details about the files:

==> Dependencies:
        Numpy
        Scipy
        Scikit-learn
        Pandas
        Surprise (link: http://surpriselib.com/)

==> The files 'sol.py' and 'driver.py' implements the proposed algorithm. Both these files and the folder 'Data' should be in the same directory while running the programme.
Just run the driver.py file and it'll show the MAE and RMSE on this dataset calculated using the proposed algorithm.

==> The files 'createClassificationFiles.py', 'makeClassifierTrainingTestingFiles.py', 'ELM.py', 'MLP.py', 'SVM_classifier.py' implements the cold-start based classification algorithm.[1]
They need data files of the folder 'FlickscoreData-26oct2018'. 
Instructions for running:
  1.) The Data-folder 'FlickscoreData-26oct2018' and code-files should be in the same folder.
  2.) First run 'makeClassifierTrainingTestingFiles.py'. This will create some new files required by other scripts.
  3.) Then run any of the classifier files ('SVM_classifier.py', 'ELM.py', 'MLP.py') that you want.
  
 ==> The file MF.py implements the SVD Matrix Factorization Method. The file and the folder 'Data' should be in the same directory while running the script.
 Just run the script and it'll print the MAE and RMSE over 5-folds.
 
 
 
 Important Note:
  The ratings in the data file are -1, 0, 1. 
  I converted them to 1, 2, 3 respectively. So it's like ratings are on the scale of 1 to 3.
  
 
 
 References:
 [1] A simple classification based approach for addressing user and item cold-start problem (link: https://drive.google.com/file/d/0B-G5D95ALDLlSkJUZ1FTaDNwejg/view)
 
 [2] Weighting strategies for a recommender system using item clustering based on genres (link: https://www.sciencedirect.com/science/article/pii/S0957417417300404?via%3Dihub)
