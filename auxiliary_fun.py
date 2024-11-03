import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import linear_model, svm #?
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn import svm/
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
# from sklearn import  ensemble
from lifelines import KaplanMeierFitter



def model_dashboard(model_name, N_classes=2):

    if model_name=='RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) # probar sacar max_depth en version no fast y ver el timepo q lleva                                                                                 #random_stateint, RandomState instance or None, default=None 
    if model_name=='RandomForestRegressor':
        # model = RandomForestRegressor(max_depth=2, random_state=0)
        model = RandomForestRegressor(min_samples_split=10, random_state=0)        
    if model_name=='LinearRegression':
        model = linear_model.LinearRegression()
    if model_name=='QuantileRegressor':
        model = linear_model.QuantileRegressor(quantile=0.5, alpha=0, solver="highs")
    if model_name=='LogisticRegression':
        model = linear_model.LogisticRegression(max_iter=1000) 
    if model_name=='KNeighborsClassifier':
        model = KNeighborsClassifier()
    if model_name=='KNeighborsRegressor':
        model = KNeighborsRegressor()
    if model_name=='SupportVectorMachines': #y esto?
        model = svm.SVC()
    if model_name=='SupportVectorClassification(LINEAL_K)':
        model = SVC(probability=True)
    if model_name=='GradientBoostingRegressor':
        params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "squared_error",
        }
        model = GradientBoostingRegressor(**params)
    if model_name=='PassiveAggressiveRegressor':
        model = linear_model.PassiveAggressiveRegressor()
    if model_name=='LassoLars':
        model = linear_model.LassoLars()
    #VER BIEN COMO IMPLEMENTAR
    if model_name=='CoxRegression': 
        model = KaplanMeierFitter()
    return model

#function to cast data before cleaning
def d_cast(DATA, TARGET_COLUMN):
    ROSSETA = {}
    print(DATA.head())

    for column in DATA:  
        # a flag to tell when to write
        cast_flag = False
        #if there is only one unique, we save the KEY and drop the column
        if len(DATA[column].unique()) == 1:
            print(10)
            cast_flag = True
            uniqueVAL = DATA[column][0] #preparar para recuperar para el final del modelado (guardar en un df)
            id_unique = 0
            DATA = DATA.drop(column, axis=1)
            a,b = -99, -99
        else: #if we drop we cant acces the colum type 
            column_type = DATA[column].dtype
            #if the column contain string we cast it to int
            if column_type == 'object':
                cast_flag = True
                DATA, uniqueVAL, id_unique = unique_to_int(DATA, column) #guardar todo esto en un DF para recuperar
                a,b = -99, -99
            #if the column contain string we cast it to int
            if column_type == 'bool' or len(DATA[column].unique())==2:
                cast_flag = True
                DATA, uniqueVAL, id_unique = unique_to_int(DATA, column) #guardar todo esto en un DF para recuperar
                # uniqueVAL = np.array(['False', 'True']) 
                uniqueVAL = np.array([False, True]) 
                a,b = -99, -99          
            else:
                if column == TARGET_COLUMN:
                    cast_flag = True
                    DATA, uniqueVAL, id_unique = unique_to_int(DATA, TARGET_COLUMN) #with perturbado it breaks
                    model = linear_model.LinearRegression()
                    model.fit(uniqueVAL.reshape(-1, 1), id_unique.reshape(-1, 1))
                    a = model.coef_      
                    b = model.intercept_ 
        if cast_flag:
            current_column= {column : [uniqueVAL, id_unique, a, b]} 
            ROSSETA.update(current_column)
    return DATA, ROSSETA


def de_cast_PREDICTION(casted_data, columns, rosseta):
    decasted_data = casted_data.copy()
    for idx_column, column in enumerate(columns):
        #case we need to decast
        if column in rosseta.keys():
            # y=ax+b
            a = rosseta[column][2] 
            b = rosseta[column][3] 
            uniqueVAL = rosseta[column][0]
            id_unique = rosseta[column][1]
            
            
            for idx_row, y_hat in enumerate(casted_data.iloc[:,idx_column].to_numpy()):
                #identity finding
                if y_hat in id_unique:
                    idx_cast = np.where(id_unique == y_hat) 
                    # cast_input[idx] = uniqueVAL[idx_cast[0][0]] 
                    decasted_data.iloc[idx_row,idx_column] = uniqueVAL[idx_cast[0][0]] 
                #identity not found
                else:
                    if y_hat>id_unique[-1] or y_hat<0:
                        x_hat = (y_hat-b)/a 
                        aux = 2
                    else:
                        idx_search = np.searchsorted(id_unique, y_hat, side='left')
                        
                        y_1, y_2 = id_unique[idx_search-1], id_unique[idx_search]
                        x_1, x_2 = uniqueVAL[idx_search-1], uniqueVAL[idx_search]
                        t_hat = (y_hat-y_1)/(y_2-y_1)
                        x_hat = x_2*t_hat + x_1*(1-t_hat) 

                    # prediction_decasted[idx] = x_hat
                    decasted_data.iloc[idx_row,idx_column] = x_hat

    return decasted_data


#not nly used for predictions
def de_cast_PREDICTION_print(casted_data, columns, rosseta):

    decasted_data = casted_data
    for idx_column, column in enumerate(columns):
        #case we need to decast
        if column in rosseta.keys():
            # y=ax+b
            a = rosseta[column][2] 
            b = rosseta[column][3] 
            uniqueVAL = rosseta[column][0]
            id_unique = rosseta[column][1]
            
            
            for idx_row, y_hat in enumerate(casted_data.iloc[:,idx_column].to_numpy()):
                #identity finding
                if y_hat in id_unique:

                    idx_cast = np.where(id_unique == y_hat) 
                    # cast_input[idx] = uniqueVAL[idx_cast[0][0]] 
                    decasted_data.iloc[idx_row,idx_column] = uniqueVAL[idx_cast[0][0]] 
                #identity not found
                else:
                    if y_hat>id_unique[-1] or y_hat<0:
                        x_hat = (y_hat-b)/a 

                        aux = 2
                    else:
                        idx_search = np.searchsorted(id_unique, y_hat, side='left')
                        
                        y_1, y_2 = id_unique[idx_search-1], id_unique[idx_search]
                        x_1, x_2 = uniqueVAL[idx_search-1], uniqueVAL[idx_search]
                        t_hat = (y_hat-y_1)/(y_2-y_1)
                        x_hat = x_2*t_hat + x_1*(1-t_hat) 

                    # prediction_decasted[idx] = x_hat
                    decasted_data.iloc[idx_row,idx_column] = x_hat

    return decasted_data

#usarlo para castear las entradas
def cast_input(data_input, rosseta):


    columns = data_input.columns

    # casted_input = np.ones(len(data_input))*-1
    casted_input = data_input

    idx_row = 0
    for idx_column, column in enumerate(columns):

        current_value = data_input.iloc[idx_row, idx_column]
        print(column)
        print(current_value)
        
        #verify if is needed to cast <if column is in rosseta.keys()>
        if column in rosseta.keys():
            a = rosseta[column][2] 
            b = rosseta[column][3] 
            uniqueVAL = rosseta[column][0]
            id_unique = rosseta[column][1]
            
            #search if the value exist has already an ID
            if current_value in uniqueVAL:

                idx_cast = np.where(uniqueVAL == current_value) 
                casted_input.iloc[idx_row, idx_column] = id_unique[idx_cast[0][0]] 
            else:
                print('no existe en la id')

    return casted_input

def unique_to_int(data_in, column_name):

    dataF = pd.DataFrame(data_in, columns=data_in.columns)
    uniqueVAL = dataF[column_name].unique()
    id_unique = np.arange(len(uniqueVAL))


    # print('\n column name')
    # print(column_name)
    # print(uniqueVAL)

    # np.isnan(uniqueVAL, casting=)
    uniqueVAL1 = uniqueVAL[~pd.isnull(uniqueVAL)] #drop uniques nan
    uniqueVAL1 = np.sort(uniqueVAL1)
    # if column_name == 'ZB1SOCUP1':


    for idU, val in enumerate(uniqueVAL1):



        dataF[column_name] = dataF[column_name].apply(lambda x: idU if x == val else x)
        # data_in[column_name] = data_in[column_name].apply(lambda x: idU if x == val else x)
        #tengo un problema, ti convierto todos los 1 en 0 
            #si tenia 0, ahora al reconvertirlos en 1 me queda 

        #en general puedo tener el problema de castear valores que ya habia casteado
            #tengo q asegurarme que los id_unique no tengan match con uniqueVAL

    return dataF, uniqueVAL1, id_unique


def unique_to_int_reverse(data_in, column_name, uniqueVAL, id_unique):
    dataF2 = pd.DataFrame(data_in, columns=data_in.columns)
    for idU, val_to_cast in enumerate(id_unique):
        val_to_replace = uniqueVAL[idU]
        dataF2[column_name] = dataF2[column_name].apply(lambda x: val_to_replace if x == val_to_cast else x)
        
    return dataF2


def computemetrics(model, X_test, y_test):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #report fow each unique class (+ avg)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) 
    cm = confusion_matrix(y_test, y_pred)

    #Compute ROC curve and AUC
    if hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _  = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)

    #Handle missing class '1' in classification_report
    if '1' in report:
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
    else:
        precision = recall = f1 = None

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr, roc_auc)
    }