import auxiliary_fun
import Dacquisition
import DprepNcleaning
import eda
import Mbuilding
import dtale
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt 


def dyn_model_selection(file_selected=-1, column_selected=-1):
    
        ### Step 1: Data acquisition ###
    data = Dacquisition.d_acquisition(file_selected)
    data, target_column, target_type, scewed_target_col = Dacquisition.var_acquisition(data, column_selected, CHECK=True) #tema de dataNaN y dataCLEAN juntar
                                                                    #tipo requirements que se hacen en el main antes de entrar a SYSTEM
    #AUTO-CAST strings to int (if <unique == 1> we save the value as a key and drop the column for the model)
    data = auxiliary_fun.d_cast(data, target_column, target_type)

        ### Step 3.1: Exploratory Data Analyzis (MANUAL)###
    manualEDA, missing4rows = eda.ManualEDAfun(data) #data no tiene a predicted column
    print(colored('\nTable with information of the variables:', 'red', attrs=['bold']))
    print(colored(manualEDA, 'red'))
    print(colored('\nTable with information of the rows:', 'red', attrs=['bold']))
    print(colored(missing4rows, 'red'))
    print(colored('\nThe result of the number of patients is: '+str(len(data)), 'red', attrs=['bold']))
    
        ### Step 2: Data Cleaning ###    
    #min_porcentage_col if missing>10 for a column, we drop it
    data = DprepNcleaning.data_cleaning(data, min_porcentage_col = 10, min_porcentage_row = 0)

        ### Step 3.2: Exploratory Data Analyzis (AUTO)###
    dtale.show(data) #hacerle decast !!!!!
    dtale.show(open_browser=True)

        ### Step 5: Model Building       
    model_info = Mbuilding.model_shake(DATA=data, TARGET_COLUMN=target_column, TARGET_TY=target_type)
    sns.lmplot(
        data=model_info, x="Cross-validation ID", y="Score", row="Normalization method", col="Feature selection method", hue='Model name',
        palette="crest", ci=None,
        height=4, scatter_kws={"s": 50, "alpha": 1}
    )
    plt.show()
    
    return model_info


