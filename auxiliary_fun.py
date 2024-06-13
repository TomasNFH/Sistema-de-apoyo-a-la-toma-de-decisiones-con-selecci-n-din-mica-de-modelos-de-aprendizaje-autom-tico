import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier





def unique_to_int(data_in, column_name):
    dataF = pd.DataFrame(data_in, columns=data_in.columns)
    uniqueVAL = dataF[column_name].unique()
    id_unique = np.arange(len(uniqueVAL))
    for idU, val in enumerate(uniqueVAL):
        dataF[column_name] = dataF[column_name].apply(lambda x: idU if x == val else x)
    
    return dataF, uniqueVAL, id_unique

def unique_to_int_reverse(data_in, column_name, uniqueVAL, id_unique):
    dataF2 = pd.DataFrame(data_in, columns=data_in.columns)
    for idU, val_to_cast in enumerate(id_unique):
        val_to_replace = uniqueVAL[idU]
        dataF2[column_name] = dataF2[column_name].apply(lambda x: val_to_replace if x == val_to_cast else x)
        
    return dataF2

def model_dashboard(model_name, N_classes=2):
    if model_name=='RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) # probar sacar max_depth en version no fast y ver el timepo q lleva
                                                                                        #random_stateint, RandomState instance or None, default=None 
    if model_name=='RandomForestRegressor':
#         model = RandomForestRegressor(max_depth=2, random_state=0)
        model = RandomForestRegressor(min_samples_split=10, random_state=0)        
        
    if model_name=='LogisticRegression':
        model = linear_model.LogisticRegression()
        
    if model_name=='KNeighborsClassifier':
        model = KNeighborsClassifier()
        
    if model_name=='SupportVectorMachines':
#         clf = svm.SVC()
        model = svm.SVC()
        
    return model