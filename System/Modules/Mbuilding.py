from System.Modules import Fselection
import numpy as np
import pandas as pd
from System.Modules import DprepNcleaning
from sklearn.model_selection import train_test_split
from System.Auxiliary import auxiliary_fun
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
from termcolor import colored
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.utils import resample
from alive_progress import alive_bar
import time


def model_shake(DATA, TARGET_COLUMN, TARGET_TY, Fast = True):

    PROGRESS_BAR = True

    start_time = time.time()
    colors_plot = ['#309284', '#337AEF']
    fig_ROC = 0
    fig_CM = 0
    FEATURE_N = 5 #can test to change it or make it auto to find the best N
    model_return = pd.DataFrame(columns = ['Target column', 'Target type', 'Model name', 
                                           'Normalization method', 'Feature selection method', 
                                           'Features used', 'Number of splits', 'Cross-validation ID', 
                                           'Confusion matrix', 'True Positive Rate', 'False Positive Rate', 'Recall', 
                                           'F1 score', 'AUC', 'Score', 'Brier score loss'])
    NORM_FLAGS = np.array([0,1,2]) #we aplly only to train data, why? to all data?
    FEATURE_FLAGS = np.array([0,1,2]) #dont use 2 due to time
    if Fast:
        FEATURE_FLAGS = np.array([0,1])

    if TARGET_TY == 'boolean':
        model_stack = ['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier', 'SupportVectorClassification', 'GradientBoostingClassifier', 'GaussianNB']
        # NORM_FLAGS = np.array([0])
    if TARGET_TY == 'classes':
        model_stack = ['RandomForestClassifier', 'KNeighborsClassifier', 'SupportVectorClassification']   
        # NORM_FLAGS = np.array([0])
    if TARGET_TY == 'continuous':
        # model_stack = ['LinearRegression', 'RandomForestRegressor','QuantileRegressor', 'GradientBoostingRegressor', 'PassiveAggressiveRegressor', 'LassoLars', 'KNeighborsRegressor'] 
        model_stack = ['LinearRegression', 'SupportVectorMachines', 'RandomForestRegressor','QuantileRegressor', 'GradientBoostingRegressor', 'PassiveAggressiveRegressor', 'LassoLars', 'KNeighborsRegressor'] 

    NORM_FLAGS = np.array([0]) #dont use normalization just for know

    Feature_methods = ['Intrinsic method','Filter method','Wrapper method']
    Normalization_methods = ['No', 'Min-Max', 'Z-score']
    
    number_of_splits = 5
    operation_counter = 0
    number_operations = len(FEATURE_FLAGS)*(number_of_splits)*len(model_stack)*len(NORM_FLAGS)


    X = DATA.loc[:, DATA.columns != TARGET_COLUMN]
    y = DATA[TARGET_COLUMN]
    columns_X = X.columns


    ### leave one out add ###


    # breakpoint()

    ### Bootstraping ###  
    min_number_of_samples = 50
    number_of_samples = X.shape[0]
    if number_of_samples < min_number_of_samples:
        X, y = resample(X, y, n_samples=min_number_of_samples, replace=True) 

    ### Shuffle ###
    #only shuffle the data
    X_frag1, X_frag2, y_frag1, y_frag2 = train_test_split(X, y, test_size=0.3, shuffle = True)
    X = np.append(X_frag1, X_frag2, axis=0)
    y = np.append(y_frag1, y_frag2, axis=0)

    ### Step 2: Cross Validation ###   
    samples_of_valid = int(len(X)/number_of_splits)
    for shift_idx in range(number_of_splits): 
        print('\n shift_idx '+str(shift_idx))
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=samples_of_valid, random_state=0, shuffle = False)
        if TARGET_TY == 'classes' or TARGET_TY == 'boolean': #quick solve for HDwithCM, solve in general!!!
            y_train = y_train.astype(int)
            y_valid = y_valid.astype(int)
        #shift data
        X = np.roll(X, samples_of_valid, axis=0)
        y = np.roll(y, samples_of_valid, axis=0)

        ### Step 1: Feature Engeeniring ###
        IMPORTANCES_OUT = []
        CURRENT_FEATURES_OUT = []
        ALL_TRAINED_MODELS = []
        for F_FLAG in FEATURE_FLAGS:
            print('F_flag '+str(F_FLAG))
            X_trainR, current_Features, importances, indexes4valid = Fselection.F_selector(pd.DataFrame(X_train, columns = columns_X), 
                                                pd.DataFrame(y_train, columns = [TARGET_COLUMN]), 
                                                N_features=FEATURE_N, 
                                                FLAG=F_FLAG) 
            X_validR = X_valid[:, indexes4valid]
            IMPORTANCES_OUT.append(importances)
            CURRENT_FEATURES_OUT.append(current_Features)

            ### Step 3: Model selection ###
            for model_name in model_stack:
                print('model '+str(model_name))

                ### Step 4: NORMALIZATION ###
                for N_FLAG in NORM_FLAGS:
                    print('normalization '+str(N_FLAG))
                    # DprepNcleaning.data_normF(X_trainR, FLAG=N_FLAG) #arreglar 

                    operation_counter = operation_counter+1
                    if PROGRESS_BAR:
                        print('Progresion in training: '+str( round((operation_counter/number_operations)*100) )+'%, the time is: ', end='')  
                        end_time = time.time()
                        total_seconds = end_time-start_time
                        minutes = int(total_seconds // 60)
                        seconds = int(total_seconds % 60)
                        print('{minutes}:{seconds}'.format(minutes=minutes, seconds=seconds)+' minutes.')

                    ### Step 5: Model Building 
                    model = auxiliary_fun.model_dashboard(model_name)
                    # if F_FLAG==2:
                        # breakpoint()
                    model.fit(X_trainR, y_train)
                    ALL_TRAINED_MODELS.append(model)
                    prediction = model.predict(X_validR)
                    
                    # define values in case we dont estimeate them (continues case 4 example)
                    accurecy = np.nan
                    Specifity = np.nan
                    tpr = np.nan
                    fpr = np.nan
                    auc = np.nan
                    Recall = np.nan
                    F1 = np.nan
                    CoMtx = np.nan
                    brier_score = np.nan

                    if TARGET_TY == 'boolean' or TARGET_TY == 'classes':
                        classes_of_target = np.unique(y)
                        CoMtx = confusion_matrix(y_valid, prediction, labels=list(classes_of_target))  #ADD LABELS TO MATRIX (MAKE IT DF)
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
                            # breakpoint()
                            metrics_result = auxiliary_fun.computemetrics(model, X_validR, y_valid)
                            tpr = metrics_result['roc_curve'][1]
                            fpr = metrics_result['roc_curve'][0]
                            auc = metrics_result['roc_curve'][2]

                            ### brier score loss ###
                            prediction_proba = model.predict_proba(X_validR)
                            prediction_proba_positive_clase = prediction_proba[:,1] 
                            brier_score = brier_score_loss(y_valid, prediction_proba_positive_clase)
                    model_return.loc[len(model_return.index)] = [TARGET_COLUMN, TARGET_TY, model_name, 
                                                                 Normalization_methods[N_FLAG], Feature_methods[F_FLAG], current_Features.values.tolist(), 
                                                                 number_of_splits, shift_idx, CoMtx, 
                                                                 tpr, fpr, Recall, F1, auc, model.score(X_validR, y_valid), brier_score] 
        breakpoint()
    feature_data = pd.DataFrame([]) 
    Feature_methods = ['Intrinsic method','Filter method','Wrapper method']
    for idx in range(len(IMPORTANCES_OUT)):
        list_of_tuples = list(zip(CURRENT_FEATURES_OUT[idx], IMPORTANCES_OUT[idx]))
        aux = pd.DataFrame(list_of_tuples, columns=['Feature', 'Score'])
        aux.insert(0, 'Feature method', Feature_methods[idx]) 
        feature_data = pd.concat([feature_data,aux],ignore_index=True) 
  
    
    if Fast == False:
        fig_features, (ax1, ax2, RandomForestClassifierax3) = plt.subplots(3, 1)
        ax3.bar(feature_data[feature_data['Feature method']==Feature_methods[2]]['Feature'], feature_data[feature_data['Feature method']==Feature_methods[2]]['Score'], color=colors_plot[0])
        ax3.set_ylim(0, 1)
    else: fig_features, (ax1, ax2) = plt.subplots(2, 1)

    ax1.bar(feature_data[feature_data['Feature method']==Feature_methods[0]]['Feature'], feature_data[feature_data['Feature method']==Feature_methods[0]]['Score'], color=colors_plot[0])
    ax1.set_ylim(0, 1)
    ax2.bar(feature_data[feature_data['Feature method']==Feature_methods[1]]['Feature'], feature_data[feature_data['Feature method']==Feature_methods[1]]['Score'], color = colors_plot[0])
    ax2.set_ylim(0, 1)
    plt.draw()
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    fig_features.tight_layout()   


    if TARGET_TY == 'boolean':
        print(colored('\nTable with information of scores of the models:', 'green', attrs=['bold']))
        print(colored(model_return[['Target column', 'Target type', 'Model name', 'Normalization method', 'Feature selection method', 'Number of splits', 'True Positive Rate', 'False Positive Rate', 'Recall', 'F1 score', 'Score', 'Brier score loss']].sort_values(by=['Score'], ascending=False).head(20), 'green'))

        print(colored('\nThe results for the best model (based in Score):', 'green', attrs=['bold']))
        MAX_idx = model_return['Score'].idxmax()
        best_model_res = model_return.iloc[MAX_idx]

        model_idx = 0
        fig_CM, axes = plt.subplots(1, len(model_return['Model name'].unique()), sharey='row')
        max_curves_per_model = 1
        number_of_models = len(model_return['Model name'].unique())
        grouped_by_model = model_return.groupby('Model name')
        for model_name_loop in model_return['Model name'].unique():
            curve_id = 0
            current_model_data = grouped_by_model.get_group(model_name_loop)
            current_model_data = current_model_data.sort_values(by=['AUC','Score','F1 score'], ascending=False)
            for index, row in current_model_data.iterrows():
                if curve_id < max_curves_per_model:

                    disp = ConfusionMatrixDisplay(row['Confusion matrix'],
                                                display_labels=list(classes_of_target))
                    disp.plot(ax=axes[model_idx], xticks_rotation=45)
                    disp.ax_.set_title(model_name_loop, rotation = 15)
                    disp.im_.colorbar.remove()
                    disp.ax_.set_xlabel('')


                curve_id = curve_id+1
            model_idx = model_idx+1
        plt.tight_layout()

        number_of_models = len(model_return['Model name'].unique())
        grouped_by_model = model_return.groupby('Model name')

        fig_ROC = plt.figure()
        model_idx = 0
        max_curves_per_model = 1
        # breakpoint()
        for model_name_loop in model_return['Model name'].unique():
            curve_id = 0
            current_model_data = grouped_by_model.get_group(model_name_loop)
            current_model_data = current_model_data.sort_values(by=['AUC','Score','F1 score'], ascending=False)
            for index, row in current_model_data.iterrows():
                if curve_id < max_curves_per_model:
                    plt.subplot(2,3,model_idx+1)
                    plt.plot(row['False Positive Rate'], row['True Positive Rate'], lw=2, label=f'(AUC={row["AUC"]:.2f})', color=colors_plot[0])
                    plt.plot([0, 1], [0, 1], color=colors_plot[1], lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive R.')
                    if model_idx == 0:
                        plt.ylabel('True Positive Rate')
                    plt.title(f'{row["Model name"]}', rotation=0)
                    plt.legend(loc="lower right")
                curve_id = curve_id+1
            model_idx = model_idx+1
        plt.tight_layout()

    if TARGET_TY == 'classes':
        print(colored('\nTable with information of scores of the models:', 'green', attrs=['bold']))
        print(colored(model_return[['Target column', 'Target type', 'Model name', 'Normalization method', 'Feature selection method', 'Number of splits', 'Score']].sort_values(by=['Score'], ascending=False).head(20), 'green'))

        print(colored('\nThe results for the best model (based in Score):', 'green', attrs=['bold']))
        MAX_idx = model_return['Score'].idxmax()
        best_model_res = model_return.iloc[MAX_idx]

    if TARGET_TY == 'continuous':
        print(colored('\nTable with information of scores of the models:', 'green', attrs=['bold']))
        print(colored(model_return[['Target column', 'Target type', 'Model name', 'Normalization method', 'Feature selection method', 'Number of splits', 'Score']].sort_values(by=['Score'], ascending=False).head(20), 'green'))


    return model_return, ALL_TRAINED_MODELS, fig_features, fig_ROC, fig_CM


