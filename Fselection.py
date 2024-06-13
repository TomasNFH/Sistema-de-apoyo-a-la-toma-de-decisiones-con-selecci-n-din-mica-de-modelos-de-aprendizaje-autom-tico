from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import numpy as np
import pandas as pd


def F_selector(X_train, y_train, X_test, N_features=5, FLAG=0):
    # print('\n INICIA')
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

        # print('EFS features')
        # print(list(efs.best_feature_names_))
        
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
            # print('Metodo correlacion')
            data = pd.concat([y_train, X_train], axis=1)
            # print('data')
            # print(data.head())
            # print('unique')
            # print(data['Domain'])
            # print(data['Domain'].unique())
            corrM = data.corr() 
            # print('correlation matrix:')
            # print(corrM)
            # print('\n')
            importances = corrM.iloc[0][1:]
            importances = importances.to_numpy()
            importances = np.abs(importances)        
        imp_sorted = np.sort(importances)

        indexes = np.ones(0, dtype = int)
        
        # print('Antes del FOR')
        for importance in imp_sorted[len(importances)-N_features:]: #q es este for?
            # print('Test, flag is')
            # print(FLAG)
            # print(np.where(importances == importance))
            idx = np.where(importances == importance)[0][0]
            indexes = np.append(indexes, int(idx))
            
        features = X_test.columns[indexes]

        X_train_reduced = X_train.iloc[:, indexes] #select columns from X
        X_test_reduced = X_test.iloc[:, indexes]

    return X_train_reduced, X_test_reduced, features 