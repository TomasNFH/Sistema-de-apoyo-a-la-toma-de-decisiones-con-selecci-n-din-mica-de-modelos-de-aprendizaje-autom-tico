from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

###Data Loading
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)


###Build and fit a classifier
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder="/tmp/autosklearn_classification_example_tmp",
)
automl.fit(X_train, y_train, dataset_name="breast_cancer")


###View the models found by auto-sklearn
print(automl.leaderboard())

