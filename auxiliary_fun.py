import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
# from sklearn import svm/
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt


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
    if model_name=='SupportVectorMachines':
        model = svm.SVC()
 
    return model

def d_cast(DATA, TARGET_COLUMN, TARGET_TYPE):
    ROSSETA = {}

    # breakpoint()
    # #cast prediction column ONLY (COULD ADD A CONTINOUS CONDITION TO ENTER)
    # DATA, uniqueVAL, id_unique = unique_to_int(DATA, TARGET_COLUMN) #with perturbado it breaks
    # auxDATA = DATA
    for column in DATA:  



        # a flag to tell when to write
        cast_flag = False
        uniqueVAL = 0
        id_unique = 0
        a = 0
        b = 0
        # breakpoint() 
        print(column)
        #if there is only one unique, we save the KEY and drop the column
        if len(DATA[column].unique()) == 1:
            print(DATA[column].unique())
            unique_val = DATA[column][0] #preparar para recuperar para el final del modelado (guardar en un df)
            DATA = DATA.drop(column, axis=1)
            breakpoint()
            print('q hago aca? <valor único>')

        else: #if we drop we cant acces the colum type    
            column_type = DATA[column].dtype
            
            #if the column contain string we cast it to int
            if column_type == 'object':
                cast_flag = True
                DATA, uniqueVAL, id_unique = unique_to_int(DATA, column) #guardar todo esto en un DF para recuperar

                model = linear_model.LinearRegression()
                #q pasa con tipo clase?
                model.fit(uniqueVAL.reshape(-1, 1), id_unique.reshape(-1, 1))
                a = model.coef_      
                b = model.intercept_ 

                
                #  print('a')
            else:
                # print('b')
                if column == TARGET_COLUMN:
                    cast_flag = True
                    # print('c')
                    # a = 1
                    #cast prediction column ONLY (COULD ADD A CONTINOUS CONDITION TO ENTER)
                    # DATA, uniqueVAL_target, id_unique_target = unique_to_int(DATA, TARGET_COLUMN) #with perturbado it breaks
                    DATA, uniqueVAL, id_unique = unique_to_int(DATA, TARGET_COLUMN) #with perturbado it breaks

                    model = linear_model.LinearRegression()
                    model.fit(uniqueVAL.reshape(-1, 1), id_unique.reshape(-1, 1))
                    a = model.coef_      
                    b = model.intercept_ 

                    # plt.plot(uniqueVAL, id_unique)
                    # y_hat = a*uniqueVAL+b 
                    # plt.plot(uniqueVAL, y_hat[0])
                    # plt.savefig('test') 
                    # breakpoint()
        if cast_flag:
            current_column= {column : [uniqueVAL, id_unique, a, b]} 
            ROSSETA.update(current_column)
        # breakpoint()
        # a=2
        # b=0
        
    # current_column= {'name': column, 'X': uniqueVAL_target, 'Y': id_unique_target} 
    
    breakpoint()
    # return DATA, uniqueVAL_target, id_unique_target
    return DATA, ROSSETA


def de_cast_PREDICTION(prediction, target_column, rosseta):

    # y=ax+b
    a = rosseta[target_column][2] 
    b = rosseta[target_column][3] 
    uniqueVAL = rosseta[target_column][0]
    id_unique = rosseta[target_column][1]
    # prediction = prediction.to_numpy()
    # prediction_decasted = np.ones(len(prediction)) 
    prediction_decasted = prediction*0 
    for idx, y_hat in enumerate(prediction.to_numpy()):
        if y_hat>=id_unique[-2] or y_hat<0:
            x_hat = (y_hat-b)/a 
            # breakpoint()
            aux = 2
        else:
            idx_search = np.searchsorted(id_unique, y_hat, side='left')
            
            y_1, y_2 = id_unique[idx_search-1], id_unique[idx_search]
            x_1, x_2 = uniqueVAL[idx_search-1], uniqueVAL[idx_search]
            t_hat = (y_hat-y_1)/(y_2-y_1)
            x_hat = x_2*t_hat + x_1*(1-t_hat) 
        # prediction_decasted[idx] = x_hat
        prediction_decasted.loc[idx] = x_hat

    return prediction_decasted

def cast_input(data_input, columns, rosseta):
    breakpoint()

def unique_to_int(data_in, column_name):
    # breakpoint()
    dataF = pd.DataFrame(data_in, columns=data_in.columns)
    uniqueVAL = dataF[column_name].unique()
    id_unique = np.arange(len(uniqueVAL))
    # breakpoint()

    # print('\n column name')
    # print(column_name)
    # print(uniqueVAL)

    # np.isnan(uniqueVAL, casting=)
    uniqueVAL1 = uniqueVAL[~pd.isnull(uniqueVAL)] #drop uniques nan
    uniqueVAL1 = np.sort(uniqueVAL1)
    # if column_name == 'ZB1SOCUP1':
    # breakpoint()
    for idU, val in enumerate(uniqueVAL1):
        
        # print(val)
        # breakpoint()
        # print(type(val))
        # if np.isnan(val):
        #     breakpoint()
        # breakpoint()

        dataF[column_name] = dataF[column_name].apply(lambda x: idU if x == val else x)
        # data_in[column_name] = data_in[column_name].apply(lambda x: idU if x == val else x)
        #tengo un problema, ti convierto todos los 1 en 0 
            #si tenia 0, ahora al reconvertirlos en 1 me queda 

        #en general puedo tener el problema de castear valores que ya habia casteado
            #tengo q asegurarme que los id_unique no tengan match con uniqueVAL
    # breakpoint()
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
    # breakpoint()
    cm = confusion_matrix(y_test, y_pred)

    #Compute ROC curve and AUC
    if hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _  = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    # breakpoint()

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