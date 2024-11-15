import auxiliary_fun
import Dacquisition
import DprepNcleaning
import eda
import Mbuilding
import dtale
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt 


def dyn_model_selection(file_selected=-1, column_selected=-1, FAST = True):
    
        ### Step 1: Data acquisition ###
    data = Dacquisition.d_acquisition(file_selected)
    data, target_column, target_type, scewed_target_col = Dacquisition.var_acquisition(data, column_selected, CHECK=True) #tema de dataNaN y dataCLEAN juntar
    # breakpoint()

        ### Step 2: Data cast ###  
    data, rosseta = auxiliary_fun.d_cast(data, target_column)
    # breakpoint()

        ### Step 3: Data Cleaning ###    
    #min_porcentage_col if missing>10 for a column, we drop it
    data, drop_col, drop_row = DprepNcleaning.data_cleaning(data, min_porcentage_col = 10, min_porcentage_row = 0)
    print(colored('\nThe result of the number of patients is: '+str(len(data)), 'red', attrs=['bold']))
    # breakpoint()

        ### Step 4: Exploratory Data Analyzis (MANUAL)###
    manualEDA, missing4rows = eda.ManualEDAfun(data) #data no tiene a predicted column
    print(colored('\nTable with information of the variables:', 'red', attrs=['bold']))
    print(colored(manualEDA, 'red'))
    print(colored('\nTable with information of the rows:', 'red', attrs=['bold']))
    print(colored(missing4rows, 'red'))
    
        ### Step 4.2: Exploratory Data Analyzis (AUTO)###
    data_decasted_aux = auxiliary_fun.de_cast_PREDICTION(data, data.columns, rosseta)  
    dtale.show(data_decasted_aux) #hacerle decast !!!!!
    dtale.show(open_browser=True)

        ### Step 5: Model Building       
    model_info, trained_models, fig1, fig2, fig3 = Mbuilding.model_shake(DATA=data, TARGET_COLUMN=target_column, TARGET_TY=target_type, Fast = True)
    sns.lmplot(
        data=model_info, x="Cross-validation ID", y="Score", row="Normalization method", col="Feature selection method", hue='Model name',
        palette="crest", ci=None,
        height=4, scatter_kws={"s": 50, "alpha": 1}
    )
    plt.show()
    
    return model_info


