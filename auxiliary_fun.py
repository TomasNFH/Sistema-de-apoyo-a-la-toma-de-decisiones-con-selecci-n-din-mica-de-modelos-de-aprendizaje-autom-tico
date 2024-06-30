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



def unique_to_int(data_in, column_name):
    breakpoint()
    dataF = pd.DataFrame(data_in, columns=data_in.columns)
    uniqueVAL = dataF[column_name].unique()
    id_unique = np.arange(len(uniqueVAL))
    for idU, val in enumerate(np.sort(uniqueVAL)):
        breakpoint()
        dataF[column_name] = dataF[column_name].apply(lambda x: idU if x == val else x)
        #tengo un problema, ti convierto todos los 1 en 0 
            #si tenia 0, ahora al reconvertirlos en 1 me queda 

        #en general puedo tener el problema de castear valores que ya habia casteado
            #tengo q asegurarme que los id_unique no tengan match con uniqueVAL
    return dataF, uniqueVAL, id_unique


def unique_to_int_reverse(data_in, column_name, uniqueVAL, id_unique):

    dataF2 = pd.DataFrame(data_in, columns=data_in.columns)
    for idU, val_to_cast in enumerate(id_unique):
        val_to_replace = uniqueVAL[idU]
        dataF2[column_name] = dataF2[column_name].apply(lambda x: val_to_replace if x == val_to_cast else x)
        
    return dataF2


def model_dashboard(model_name, N_classes=2):

    if model_name=='RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) # probar sacar max_depth en version no fast y ver el timepo q lleva                                                                                 #random_stateint, RandomState instance or None, default=None 
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

def d_cast(DATA, TARGET_COLUMN, TARGET_TYPE):
    
    #cast prediction column ONLY (COULD ADD A CONTINOUS CONDITION TO ENTER)
    # breakpoint()
    DATA, uniqueVAL, id_unique = unique_to_int(DATA, TARGET_COLUMN) #with perturbado it breaks

    for column in DATA:    
        #if there is only one unique, we save the KEY and drop the column
        if len(DATA[column].unique()) == 1:
            print('D_cast')
            print(column)
            print(DATA[column].unique())
            unique_val = DATA[column][0] #preparar para recuperar para el final del modelado (guardar en un df)
            DATA = DATA.drop(column, axis=1)
        else: #if we drop we cant acces the colum type    
            column_type = DATA[column].dtype
            #if the column contain string we cast it to int
            if column_type == 'object':
                 DATA, uniqueVAL, id_unique = unique_to_int(DATA, column) #guardar todo esto en un DF para recuperar

    return DATA


def computemetrics(model, X_test, y_test):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
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