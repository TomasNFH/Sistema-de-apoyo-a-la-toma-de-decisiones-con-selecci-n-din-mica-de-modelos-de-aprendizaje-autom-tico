from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def F_selector(X_train, y_train, N_features=5, FLAG=0):
    '''
    input: X_train, y_train - used for training
    return:
    '''
    if FLAG == 2: 
        if X.shape[1]>N_features: Mf = N_features
        else: Mf = X.shape[1]

        #Create an EFS object
        efs = EFS(estimator=LogisticRegression(),        # Use logistic regression as the classifier/estimator
                min_features=1,      # The minimum number of features to consider is 1
                max_features=Mf,      # The maximum number of features to consider is 4
                scoring='accuracy',  # The metric to use to evaluate the classifier is accuracy 
                cv=5,print_progress=True)
        # breakpoint()
        efs = efs.fit(X_train, y_train)
        breakpoint()
        # efs.finalize_fit()

        features = pd.Index(efs.best_feature_names_)
        X_reduced = X[features] 
        importancesRET = np.ones(len(features))*efs.best_score_ 

    else:
        if FLAG == 0:
            model = RandomForestRegressor()
            model.fit(X_train.to_numpy(), y_train.to_numpy().ravel())
            importances = model.feature_importances_ #puedo cambiar el impurity de Gini a otro, probar
        if FLAG == 1:
            # breakpoint()
            data = pd.concat([y_train, X_train], axis=1)
            corrM = data.corr() 
            importances = corrM.iloc[0][1:]
            importances = importances.to_numpy()
            importances = np.abs(importances)
            importances = importances[np.logical_not(np.isnan(importances))] 
        imp_sorted = np.sort(importances)
        if len(importances) <= N_features: importancesRET = imp_sorted
        else: importancesRET = imp_sorted[len(importances)-N_features:]
        indexes = np.ones(0, dtype = int)
        #we run over the N_Features more importante and finde the index of each one
        for importance in importancesRET: 
            idx = np.where(importances == importance)[0][0]
            indexes = np.append(indexes, int(idx))
        features = X_train.columns[indexes] #select the column names by the indexes
        X_reduced = X_train.iloc[:, indexes] #reduce X by the most important feature
        X_reduced = X_reduced.to_numpy() #input was DF, i need np array to model.fit()
    return X_reduced, features , importancesRET, indexes

