import Fselection
import numpy as np
import pandas as pd
import DprepNcleaning
from sklearn.model_selection import train_test_split
import auxiliary_fun
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from termcolor import colored
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


def model_shake(DATA, TARGET_COLUMN, TARGET_TY, Fast = 1):
  
    model_return = pd.DataFrame(columns = ['Target column', 'Taget type', 'Model name', 
                                           'Normalization method', 'Feature selection method', 
                                           'Features used', 'Number of splits', 'Cross-validation ID', 
                                           'Confusion matrix', 'Sensitivity', '100-Specifity', 'Recall', 
                                           'F1 score', 'Score'])
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
        # model_stack = ['LinearRegression'] #in the case 

    ### Step 1: Feature Engeeniring ###
    for F_FLAG in FEATURE_FLAGS:
    
        X = DATA.loc[:, DATA.columns != TARGET_COLUMN]
        y = DATA[TARGET_COLUMN]
        X, current_Features, importances = Fselection.F_selector(X, y, N_features=FEATURE_N, FLAG=F_FLAG) #PQ MANDO Y TAMBIEN?

        print(colored('\nThe Features used with this model where', 'green', attrs=['bold']))
        print(colored(list(current_Features), 'green'))
        print(colored(list(importances), 'green'))

        # std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=current_Features)
        fig, ax = plt.subplots()
        # forest_importances.plot.bar(yerr=std, ax=ax)
        forest_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

        #BOOTSTRAPPING
        min_number_of_samples = 50
        number_of_samples = 100
        if number_of_samples >= min_number_of_samples:
            X, y = resample(X, y, n_samples=number_of_samples, replace=True) 
        else:
            X, y = resample(X, y, n_samples=min_number_of_samples, replace=True)
        # breakpoint()
        # #bootstraping in we have not enough samples
        # # the idea is to use bootstraping in the training data
        # # BOOTSTRAPING PARA Q SEA EFECTIVO LO TENGO Q HACER MUCHAS VECES
        #OTRA IDEA ES USAR SIEMPRE BOOTSREPING SIN IMPORTAR EL NRO DE MUESTRAS 
        # min_number_of_traing_samples = 100
        # if len(X) <= min_number_of_traing_samples:
        #     X, y = resample(X, y, n_samples=min_number_of_samples,replace=True) 
        # breakpoint()

        ### Step 2: Cross Validation ###
        #suffle the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True)
        X = np.append(X_train, X_test,axis = 0)
        y = np.append(y_train, y_test,axis = 0)
        number_of_splits = 5  
        samples_of_test = int(len(X)/number_of_splits)
        for shift_idx in range(number_of_splits): 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=samples_of_test, random_state=0, shuffle = False)
            #shift data
            X = np.roll(X, samples_of_test, axis=0)
            y = np.roll(y, samples_of_test, axis=0)
            # breakpoint()
            ### Step 3: Model selection ###
            for model_name in model_stack:
                # print(model_name)
                if model_name == 'SupportVectorMachines': #LR dont work with norm 2 (boolean)
                    NORM_FLAGS = np.array([0]) 

                ### Step 4: NORMALIZATION ###
                for N_FLAG in NORM_FLAGS:              
                    # DATA = DprepNcleaning.data_normF(DATA, FLAG=N_FLAG)  #NO LE ESTOY HACIENDO NORM A NADA
                    ### Step 5: Model Building 
                    model = auxiliary_fun.model_dashboard(model_name)
                    # print('/n/n\n\n')
                    # print((X_train))
                    # print((y_train))
                    model.fit(X_train, y_train)
                    prediction = model.predict(X_test)

                    # # bootstrap predictions (REVISAR BIEN ANTES DE IMPLEMENTAR BIEN)
                    # accuracy_bootstrap = []
                    # n_iterations = 1000
                    # for i in range(n_iterations):
                    #     X_bs, y_bs = resample(X_train, y_train, replace=True)
                    #     # make predictions to a bootstrap 
                    #     y_hat = model.predict(X_bs)
                    #     # evaluate model
                    #     score = accuracy_score(y_bs, y_hat)
                    #     accuracy_bootstrap.append(score)
                    # # de esto tengo q sacar info de la distribucion de probabilidad
                    
                    ###
                    accurecy = np.nan
                    Specifity = np.nan
                    Recall = np.nan
                    F1 = np.nan
                    CoMtx = np.nan
                    ###
                    if TARGET_TY == 'boolean' or TARGET_TY == 'classes':
                        # breakpoint()
                        # classes_of_target = y.unique()  #COULD BE A PRLOBLEM
                        classes_of_target = np.unique(y)
                        CoMtx = confusion_matrix(y_test, prediction, labels=list(classes_of_target))  #ADD LABELS TO MATRIX (MAKE IT DF)
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
                    model_return.loc[len(model_return.index)] = [TARGET_COLUMN, TARGET_TY, model_name, N_FLAG, F_FLAG, current_Features, number_of_splits, shift_idx, CoMtx, accurecy, Specifity, Recall, F1, model.score(X_test, y_test)] 
    #show results
    if TARGET_TY == 'boolean':
        print(colored('\nTable with information of scores of the models:', 'green', attrs=['bold']))
        print(colored(model_return[['Target column', 'Taget type', 'Model name', 'Normalization method', 'Feature selection method', 'Number of splits', 'Sensitivity', '100-Specifity', 'Recall', 'F1 score', 'Score']].sort_values(by=['Score'], ascending=False).head(20), 'green'))

        print(colored('\nThe results for the best model (based in Score):', 'green', attrs=['bold']))
        MAX_idx = model_return['Score'].idxmax()
        best_model_res = model_return.iloc[MAX_idx]
        # print(colored('\nConfusion matrix:', 'green', attrs=['bold']))
        # print(colored(pd.DataFrame(best_model_res['Confusion matrix'], columns=classes_of_target, index=classes_of_target), 'green'))
        disp = ConfusionMatrixDisplay(confusion_matrix = best_model_res['Confusion matrix'], display_labels=list(classes_of_target))    
        disp.plot() 
        plt.title('Confusion matrix')
        plt.show()
        # print(colored('\nThe Features used with this model where', 'green', attrs=['bold']))
        # print(colored(list(best_model_res['Features used']), 'green'))
        

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


