import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from multiprocessing import cpu_count
import sys

best_scores = np.empty(7)
test_scores = np.empty(7)
C_linear = [0.1, 0.5, 1, 5, 10, 50, 100]
C_poly = [0.1, 1, 3]
C_rbf = [0.1, 0.5, 1, 5, 10, 50, 100]
C_lr = [0.1, 0.5, 1, 5, 10, 50, 100]
degree_linear = [4, 5, 6]
gamma_poly = [0.1, 0.5]
gamma_rbf = [0.1, 0.5, 1, 3, 6, 10]
n_neighbors = np.arange(1, 50)
leaf_size = np.arange(5, 60, 5)
max_depth = np.arange(1, 50)
min_samples_split = np.arange(2, 10)

param_svm_linear = {"C": C_linear,
                    # "degree": degree_linear,
                    # "gamma": [0.1, 0.5, 1]
                    }

param_svm_poly = {"C": C_poly,
                  "degree": degree_linear,
                  "gamma": gamma_poly}

param_svm_rbf = {"C": C_rbf,
                 # "degree": degree_linear,
                 "gamma": gamma_rbf}

params_lr = {"C": C_lr}

params_nn = {"n_neighbors": n_neighbors,
             "leaf_size": leaf_size,
             }

params_dt = {"max_depth": max_depth,
             "min_samples_split": min_samples_split}

params = [param_svm_linear,
          param_svm_poly,
          param_svm_rbf,
          params_lr,
          params_nn,
          params_dt,
          params_dt]

folds = StratifiedShuffleSplit(5, train_size=.6, test_size=.4)
lr = LogisticRegression()
nn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svm_linear = SVC(kernel="linear")
svm_poly = SVC(kernel="poly")
svm_rbf = SVC(kernel="rbf")

models = [svm_linear,
          svm_poly,
          svm_rbf,
          lr,
          nn,
          dt,
          rf]

input_file = sys.argv[1].lower()
output_file = sys.argv[2].lower()
data = np.asarray(pd.read_csv(input_file, header=0).values)
X = data[:, :-1]
y = data[:, -1]

names = ["svm_linear",
         "svm_polynomial",
         "svm_rbf",
         "logistic",
         "knn",
         "decision_tree",
         "random_forest"]

for model, param, name in zip(models, params, names):
    grid = GridSearchCV(model, param, cv=folds, n_jobs=cpu_count() - 1, verbose=0, return_train_score=True,
                        scoring="accuracy")
    print("TRAINING %s" % name)
    grid.fit(X, y)
    best_score = grid.best_score_
    scores = pd.DataFrame(grid.cv_results_)
    mean_test_score = np.mean(scores["mean_test_score"].values)

    with open(output_file, "a") as file:
        file.write("{}, {}, {}\n".format(name, best_score, mean_test_score))
