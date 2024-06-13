import Fselection
import numpy as np
import pandas as pd
import DprepNcleaning
from sklearn.model_selection import train_test_split
import auxiliary_fun
from sklearn.metrics import confusion_matrix



def model_shake(DATA, PREDICTED_CL, TARGET_TY, Fast = 1):
    
    print('target_type is '+str(TARGET_TY))
    
    model_return = pd.DataFrame(columns = ['Target column', 'Taget type', 'Model name', 'Normalization method', 'Feature selection method', 'Features used', 'Test size', 'Confusion matrix', 'Sensitivity', '100-Specifity', 'Recall', 'F1 score', 'Score'])
        
#     NORM_FLAGS = np.array([1]) # NFLAG = 0 se rompe con RF 
    NORM_FLAGS = np.array([0,1,2])
    FEATURE_FLAGS = np.array([0,1,2]) #dont use 2 due to time
    if Fast:
        FEATURE_FLAGS = np.array([0,1])
    FEATURE_N = 5 #can test to change it or make it auto to find the best N
    TS_center = 0.33
    TS_delta = 0.2
    TS_N = 10
    test_size_array = np.linspace(TS_center-TS_delta,TS_center+TS_delta, TS_N)
    random_state_var = 42

#     Lineal regression 
# bayes network
    
    if TARGET_TY == 'boolean':
        model_stack = ['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier', 'SupportVectorMachines']
        NORM_FLAGS = np.array([0,1])
    if TARGET_TY == 'classes':
        model_stack = ['RandomForestClassifier', 'KNeighborsClassifier', 'SupportVectorMachines']   
        NORM_FLAGS = np.array([0])
    if TARGET_TY == 'continuous':
        model_stack = ['RandomForestRegressor', 'SupportVectorMachines'] 
        
    for model_name in model_stack:

        if model_name == 'SupportVectorMachines': #LR dont work with norm 2 (boolean)
            NORM_FLAGS = np.array([0]) 
        
        print('\n\nCurrent model is: ')
        print(model_name)
        
        ### Step 2 ###
        #NORMALIZATION    
        for N_FLAG in NORM_FLAGS: 
            
            print('\n\nCurrent norm is is: ')
            print(N_FLAG)
            
            data_N = DprepNcleaning.data_normF(DATA, FLAG=N_FLAG) #rompe RFC
#             data_N = DATA
    
            #no es siempre necesario o si, norm siempre mejora los resultados de prediccion
            #agregar una matriz con identidades para desnormalizar 

            #TRAIN AND TEST DEFINITION
            for test_size_var in test_size_array:
                X = data_N.loc[:, data_N.columns != PREDICTED_CL]
                y = data_N[PREDICTED_CL]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_var, random_state=random_state_var)

                ### Step 4: Feature Engeeniring ###
                for F_FLAG in FEATURE_FLAGS:
#                     print('\n\nCurrent feature is is: ')
#                     print(F_FLAG)
                    X_trainR, X_testR, current_Features = Fselection.F_selector(X_train, y_train, X_test, N_features=5, FLAG=F_FLAG)

                    ### Step 5: Model Building 
                    N_classes = len(y_train.unique())
                    model = auxiliary_fun.model_dashboard(model_name)
                    model.fit(X_trainR, y_train)
                    prediction = model.predict(X_testR)
            
                    ###
                    accurecy = np.nan
                    Specifity = np.nan
                    Recall = np.nan
                    F1 = np.nan
                    CoMtx = np.nan
                    ###
                    
                    if TARGET_TY == 'boolean' or TARGET_TY == 'classes':
                        CoMtx = confusion_matrix(y_test, prediction) #ADD LABELS TO MATRIX (MAKE IT DF)
                        accurecy = np.sum(np.diag(CoMtx))/len(prediction)
                        accurecy = accurecy*100
                        if TARGET_TY == 'boolean':
                            TP = CoMtx[0,0]
                            TP_FN = np.sum(CoMtx[0,:])
                            Recall = TP/TP_FN
                            TN = CoMtx[1,1]
                            TN_FP = CoMtx[1,:]
                            Specifity = TN/np.sum(TN_FP)
                            Specifity = (1-Specifity)*100
                            F1 = 2*(accurecy*Recall)/((accurecy+Recall))
                    model_return.loc[len(model_return.index)] = [PREDICTED_CL, TARGET_TY, model_name, N_FLAG, F_FLAG, current_Features, test_size_var, CoMtx, accurecy, Specifity, Recall, F1, model.score(X_testR, y_test)] 
                   
    return model_return