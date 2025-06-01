from System.Auxiliary import auxiliary_fun
from System.Modules import Dacquisition
from System.Modules import DprepNcleaning
from System.Modules import eda
from System.Modules import Mbuilding
import dtale
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split


def dyn_model_selection(data = pd.DataFrame(), file_selected=-1, column_selected=-1, FAST = True, PLOT = True, local_file = False):
    
    itcn = True #true menu/ false digit

    Server = False
    if Server:
        itcn = False
        local_file = False

        ### Step 1: Data acquisition ###
    if len(data)==0:
        data = Dacquisition.d_acquisition(file_selected, itcn)
    data, target_column, target_type, scewed_target_col = Dacquisition.var_acquisition(data, column_selected, CHECK=True, interaction=itcn) #tema de dataNaN y dataCLEAN juntar

        ### Step 2: Data cast ###  
    data, rosseta = auxiliary_fun.d_cast(data, target_column)

        ### Step 3: Data Cleaning ### 
    #min_porcentage_col if missing>10 for a column, we drop it
    data, drop_col, drop_row = DprepNcleaning.data_cleaning(data, min_porcentage_col = 10, min_porcentage_row = 0)
    print(colored('\nThe result of the number of patients is: '+str(len(data)), 'red', attrs=['bold']))

        ### Step 4: Exploratory Data Analyzis (MANUAL)###
    manualEDA, missing4rows = eda.ManualEDAfun(data) #data no tiene a predicted column
    print(colored('\nTable with information of the variables:', 'red', attrs=['bold']))
    print(colored(manualEDA, 'red'))
    print(colored('\nTable with information of the rows:', 'red', attrs=['bold']))
    print(colored(missing4rows, 'red'))
    
        ### Step 4.2: Exploratory Data Analyzis (AUTO)###
    # data_decasted_aux = auxiliary_fun.de_cast_PREDICTION(data, data.columns, rosseta)  
    # dtale.show(data_decasted_aux) #hacerle decast !!!!!
    # dtale.show(open_browser=True)

        ### Step 5: Model Building  

    #test data extraction
    X = data.loc[:, data.columns != target_column]
    y = data[target_column]
    X_data, X_test, y_data, y_test = train_test_split(X, y, test_size=0.1, shuffle = True)
    data = pd.concat([y_data, X_data], axis=1)

    model_info, _, fig_feat, fig_cm, fig_roc, fig_score = Mbuilding.model_shake(DATA=data, X_TEST=X_test.to_numpy(),  Y_TEST=y_test.to_numpy(), TARGET_COLUMN=target_column, TARGET_TY=target_type, Fast = FAST)
    
    if local_file: 
        # plt.savefig('Output/accuracy.png')
        fig_feat.savefig('Output/features.png')
        if fig_cm!=0: fig_cm.savefig('Output/AUC.png')
        if fig_roc!=0: fig_roc.savefig('Output/MConf.png')
        if fig_score!=0: fig_score.savefig('Output/Score.png')
    if PLOT: plt.show()

    return model_info


