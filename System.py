import auxiliary_fun
import Dacquisition
import DprepNcleaning
import eda
import Mbuilding
# from ydata_profiling import ProfileReport 
import dtale
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt 




def SystemFUN(file_selected=-1, column_selected=-1):
    
    # ### Step 1 ###
    dataADQ = Dacquisition.Data_adq(file_selected)
    data, predicted_column, target_type, unique4re_cast, id_unique, scewed_target_col = Dacquisition.variable_adq(dataADQ, column_selected, check=True) #tema de dataNaN y dataCLEAN juntar

    ### Step 2 ###                                                                    #tipo requirements que se hacen en el main antes de entrar a SYSTEM
    #AUTO-CAST strings to int (if <unique == 1> we save the value as a key and drop the column for the model)
    data = auxiliary_fun.AutoCast(data)
    data = DprepNcleaning.data_cleaning(data, min_porcentage_col = 10, min_porcentage_row = 0)

        ### Step 3: Exploratory Data Analyzis ###
    ###Manual###
    manualEDA, missing4rows = eda.ManualEDAfun(data) #data no tiene a predicted column

    print(colored('\nTable with information of the variables:', 'red', attrs=['bold']))
    print(colored(manualEDA, 'red'))
    print(colored('\nTable with information of the rows:', 'red', attrs=['bold']))
    print(colored(missing4rows, 'red'))
    print(colored('\nThe result of the number of patients is: '+str(len(data)), 'red', attrs=['bold']))
    
    ###Auto###
    #Dtale
    # dtale.show(data) #hacerle decast !!!!!
    # dtale.show(open_browser=True)

    #Pandas profiling
    # profile = ProfileReport(dataADQ, title="Profiling Report")
    # print(profile)


        ### Step 5: Model Building       
    model_info = Mbuilding.model_shake(DATA=data, PREDICTED_CL=predicted_column, TARGET_TY=target_type)

    sns.lmplot(
        data=model_info, x="Number of splits", y="Score", row="Normalization method", col="Feature selection method", hue='Model name',
        palette="crest", ci=None,
        height=4, scatter_kws={"s": 50, "alpha": 1}
    )
    plt.show()
    
    return model_info


