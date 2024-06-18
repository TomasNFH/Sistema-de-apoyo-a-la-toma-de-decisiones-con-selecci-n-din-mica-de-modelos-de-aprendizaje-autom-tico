import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
# from sklearn import svm/
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
        # model = RandomForestRegressor(max_depth=2, random_state=0)
        model = RandomForestRegressor(min_samples_split=10, random_state=0)        
    
    if model_name=='LinearRegression':
        model = linear_model.LinearRegression()

    if model_name=='LogisticRegression':
        model = linear_model.LogisticRegression()
        
    if model_name=='KNeighborsClassifier':
        model = KNeighborsClassifier()
        
    if model_name=='SupportVectorMachines':
        model = svm.SVC()


        
    return model

def AutoCast(DATA):
    for column in DATA:
        # print('\n')
        # print(column)
        
        #if there is only one unique, we save the KEY and drop the column
        if len(DATA[column].unique()) == 1:
            # print('UNIQUE DROP')
            unique_val = DATA[column][0] #preparar para recuperar para el final del modelado (guardar en un df)
            DATA = DATA.drop(column, axis=1)
        else: #if we drop we cant acces the colum type    
            column_type = DATA[column].dtype
            #if the column contain string we cast it to int
            if column_type == 'object':
                #  print('CASTTTTTT')
                 DATA, uniqueVAL, id_unique = auxiliary_fun.unique_to_int(DATA, column) #guardar todo esto en un DF para recuperar

    return DATA