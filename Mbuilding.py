import Fselection
import numpy as np
import pandas as pd
import DprepNcleaning
from sklearn.model_selection import train_test_split
import auxiliary_fun
from sklearn.metrics import confusion_matrix
from termcolor import colored
import matplotlib.pyplot as plt 
import seaborn as sns


def model_shake(DATA, PREDICTED_CL, TARGET_TY, Fast = 1):

    number_of_splits = 5    
    model_return = pd.DataFrame(columns = ['Target column', 'Taget type', 'Model name', 'Normalization method', 'Feature selection method', 'Features used', 'Number of splits', 'Confusion matrix', 'Sensitivity', '100-Specifity', 'Recall', 'F1 score', 'Score'])
    NORM_FLAGS = np.array([0,1,2])
    FEATURE_FLAGS = np.array([0,1,2]) #dont use 2 due to time
    if Fast:
        FEATURE_FLAGS = np.array([0,1])
    FEATURE_N = 5 #can test to change it or make it auto to find the best N
    
    if TARGET_TY == 'boolean':
        model_stack = ['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier', 'SupportVectorMachines']
        NORM_FLAGS = np.array([0])
    if TARGET_TY == 'classes':
        model_stack = ['RandomForestClassifier', 'KNeighborsClassifier', 'SupportVectorMachines']   
        NORM_FLAGS = np.array([0])
    if TARGET_TY == 'continuous':
        model_stack = ['LinearRegression', 'SupportVectorMachines', 'RandomForestRegressor'] 
        # model_stack = ['LinearRegression'] 

    ### Step 1: Feature Engeeniring ###
    for F_FLAG in FEATURE_FLAGS:
        X = DATA.loc[:, DATA.columns != PREDICTED_CL]
        y = DATA[PREDICTED_CL]
        X_Reduced, current_Features, importances = Fselection.F_selector(X, y, N_features=FEATURE_N, FLAG=F_FLAG)
        X_train, X_test, y_train, y_test = train_test_split(X_Reduced, y, test_size=number_of_splits, random_state=0, shuffle = True)#data split ADD CV

        ### Step 2: Model selection ###
        for model_name in model_stack:
            if model_name == 'SupportVectorMachines': #LR dont work with norm 2 (boolean)
                NORM_FLAGS = np.array([0]) 
            ### Step 3: NORMALIZATION ###
            for N_FLAG in NORM_FLAGS:              
                DATA = DprepNcleaning.data_normF(DATA, FLAG=N_FLAG) 
                ### Step 5: Model Building 
                model = auxiliary_fun.model_dashboard(model_name)
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)
                ###
                accurecy = np.nan
                Specifity = np.nan
                Recall = np.nan
                F1 = np.nan
                CoMtx = np.nan
                ###
                if TARGET_TY == 'boolean' or TARGET_TY == 'classes':
                    classes_of_target = y.unique()  #COULD BE A PRLOBLEM
                    CoMtx = confusion_matrix(y_test, prediction, labels=list(classes_of_target))  #ADD LABELS TO MATRIX (MAKE IT DF)
                    accurecy = np.sum(np.diag(CoMtx))/len(prediction)
                    accurecy = accurecy*100
                    # breakpoint()
                    if TARGET_TY == 'boolean':
                        TP = CoMtx[0,0]
                        TP_FN = np.sum(CoMtx[0,:])
                        Recall = TP/TP_FN
                        TN = CoMtx[1,1]
                        TN_FP = CoMtx[1,:]
                        Specifity = TN/np.sum(TN_FP)
                        Specifity = (1-Specifity)*100
                        F1 = 2*(accurecy*Recall)/((accurecy+Recall))
                model_return.loc[len(model_return.index)] = [PREDICTED_CL, TARGET_TY, model_name, N_FLAG, F_FLAG, current_Features, number_of_splits, CoMtx, accurecy, Specifity, Recall, F1, model.score(X_test, y_test)] 

    #show results
    if TARGET_TY == 'boolean':
        print(colored('\nTable with information of scores of the models:', 'green', attrs=['bold']))
        print(colored(model_return[['Target column', 'Taget type', 'Model name', 'Normalization method', 'Feature selection method', 'Number of splits', 'Sensitivity', '100-Specifity', 'Recall', 'F1 score', 'Score']].sort_values(by=['Score'], ascending=False).head(20), 'green'))

        print(colored('\nThe results for the best model (based in Score):', 'green', attrs=['bold']))
        MAX_idx = model_return['Score'].idxmax()
        best_model_res = model_return.iloc[MAX_idx]
        print(colored('\nConfusion matrix:', 'green', attrs=['bold']))
         
        print(colored(pd.DataFrame(best_model_res['Confusion matrix'], columns=classes_of_target, index=classes_of_target), 'green'))
        print(colored('\nThe Features used with this model where', 'green', attrs=['bold']))
        print(colored(list(best_model_res['Features used']), 'green'))

        graph = sns.lmplot(model_return, x= "100-Specifity", y="Sensitivity", hue='Model name', col="Model name", palette="crest", ci=None,height=4, scatter_kws={"s": 100, "alpha": 1})
        graph.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
        graph.fig.suptitle('ROC for the models')
        plt.xlim(0,100)
        plt.ylim(0,100)
        plt.show()
    if TARGET_TY == 'classes':
        print(colored('\nTable with information of scores of the models:', 'green', attrs=['bold']))
        print(colored(model_return[['Target column', 'Taget type', 'Model name', 'Normalization method', 'Feature selection method', 'Number of splits', 'Sensitivity', 'Score']].sort_values(by=['Score'], ascending=False).head(20), 'green'))

        print(colored('\nThe results for the best model (based in Score):', 'green', attrs=['bold']))
        MAX_idx = model_return['Score'].idxmax()
        best_model_res = model_return.iloc[MAX_idx]
        print(colored('\nConfusion matrix:', 'green', attrs=['bold']))
         
        print(colored(pd.DataFrame(best_model_res['Confusion matrix (where the true values are in the rows and the predicted in the columns)'], columns=classes_of_target, index=classes_of_target), 'green'))
        print(colored('\nThe Features used with this model where', 'green', attrs=['bold']))
        print(colored(list(best_model_res['Features used']), 'green'))

    if TARGET_TY == 'continuous':
        print(colored('\nTable with information of scores of the models:', 'green', attrs=['bold']))
        print(colored(model_return[['Target column', 'Taget type', 'Model name', 'Normalization method', 'Feature selection method', 'Number of splits', 'Score']].sort_values(by=['Score'], ascending=False).head(20), 'green'))

    return model_return


