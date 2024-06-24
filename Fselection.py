from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def F_selector(X, y, N_features=5, FLAG=0):

    #split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    if FLAG == 2: #N_features is only for FLAG == 0 and 2
        efs = EFS(
            estimator=RandomForestClassifier(
            n_estimators=3, 
            random_state=0), 
            min_features=1,
            max_features=N_features,
            scoring='roc_auc',
            cv=2,
        )
        efs = efs.fit(X_train, y_train)
        features = list(efs.best_feature_names_)
        
        X_train_t = efs.transform(X_train)
        X_train_reduced = X_train[list(efs.best_feature_names_)]
        X_test_reduced = X_test[list(efs.best_feature_names_)]
    else:
        if FLAG == 0:
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            importances = model.feature_importances_ #puedo cambiar el impurity de Gini a otro, probar
        if FLAG == 1:
            data = pd.concat([y_train, X_train], axis=1)
            corrM = data.corr() 
            importances = corrM.iloc[0][1:]
            importances = importances.to_numpy()
            importances = np.abs(importances)
        imp_sorted = np.sort(importances)
        importancesRET = imp_sorted
        indexes = np.ones(0, dtype = int)
        #we run over the N_Features more importante and finde the index of each one
        for importance in imp_sorted[len(importances)-N_features:]: 
            idx = np.where(importances == importance)[0][0]
            indexes = np.append(indexes, int(idx))
        features = X_test.columns[indexes] #select the column names by the indexes
        X_reduced = X.iloc[:, indexes] #reduce X by the most important feature

    return X_reduced, features , importancesRET

