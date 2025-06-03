from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def F_selector(X_train, y_train, N_features=5, FLAG=0):
    """
    Feature selection function.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or pd.DataFrame): Training labels.
        N_features (int): Number of top features to select.
        FLAG (int): Selection method:
            0 - Random Forest feature importance
            1 - Correlation-based selection
            2 - Exhaustive Feature Selector (EFS) with Logistic Regression

    Returns:
        X_reduced (np.ndarray): Reduced feature set.
        selected_features (pd.Index): Names of selected features.
        importances_selected (np.ndarray): Importance values of selected features.
        selected_indexes (np.ndarray): Indexes of selected features.
    """
    if FLAG == 2: 
        # Limit maximum number of features to those present in the dataset
        Mf = min(X_train.shape[1], N_features)

        #Create an EFS object
        efs = EFS(estimator=LogisticRegression(),       
                min_features=1,      
                max_features=Mf,     
                scoring='accuracy',  
                cv=5, print_progress=True)
        efs = efs.fit(X_train.to_numpy(), y_train.to_numpy().ravel())
        
        # Get selected feature indexes from their names
        indexes = list(map(int, efs.best_feature_names_))
        features = X_train.columns[indexes]
        X_reduced = X_train.iloc[:, indexes]
        importancesRET = np.ones(len(features))*efs.best_score_ 

    else:
        if FLAG == 0:
            # Feature importance via Random Forest
            model = RandomForestRegressor()
            model.fit(X_train.to_numpy(), y_train.to_numpy().ravel())
            importances = model.feature_importances_ # I can change the Gini impurity to another criterion, worth testing
        if FLAG == 1:
            # Feature importance via correlation with target
            data = pd.concat([y_train, X_train], axis=1)
            corrM = data.corr() 
            importances = corrM.iloc[0][1:]
            importances = importances.to_numpy()
            importances = np.abs(importances)
            importances = importances[np.logical_not(np.isnan(importances))] 
            
        # Sort importances and select the top N    
        imp_sorted = np.sort(importances)
        if len(importances) <= N_features: importancesRET = imp_sorted
        else: importancesRET = imp_sorted[len(importances)-N_features:]
        
        # Retrieve indexes corresponding to selected importance values
        indexes = np.ones(0, dtype = int)
        for importance in importancesRET: 
            idx = np.where(importances == importance)[0][0]
            indexes = np.append(indexes, int(idx))

        features = X_train.columns[indexes] 
        X_reduced = X_train.iloc[:, indexes] 
        X_reduced = X_reduced.to_numpy()
    return X_reduced, features , importancesRET, indexes

