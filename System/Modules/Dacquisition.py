import os
import pandas as pd
from System.Auxiliary import auxiliary_fun
import static_frame as sf
from termcolor import colored
import numpy as np
import cutie



def d_acquisition(FILE_SELECTED = -1, interaction = True):
    """
    Loads a dataset file from the 'Input' directory using an interactive menu or index selection.
    
    Parameters:
        FILE_SELECTED (int): Index of file to load. If -1, prompts user.
        interaction (bool): Whether to use interactive selection or manual input.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    files = []
    names = [] #just the name to print
    # for dirname, _, filenames in os.walk(r'C:\Users\tomas\Pasantia\Input'):
    for dirname, _, filenames in os.walk('Input'):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))    
            names.append(filename) 

    if interaction:
        # Menu implementation
        if FILE_SELECTED == -1:
            print(colored('The possible files to load are: \n', 'green', attrs=['bold']))
            captions = []
            name = names[cutie.select(names, caption_indices=captions, selected_index=0)]
            res_list = [i for i, value in enumerate(names) if value == name]
            print(int(res_list[0]))
            FILE_SELECTED = int(res_list[0])

    else:
        # Digit implementation
        if FILE_SELECTED == -1: #if ==-1 ask for the idx
            print(colored('The possible files to load are: \n', 'black', attrs=['bold']))

            for file_idx,name in enumerate(names):
                print('----> File number '+str(file_idx)+': '+str(name)) 
            FILE_SELECTED = int(input("\n\nDigit the file number: "))
        print('File selected is: '+str(files[FILE_SELECTED]))

    # data type verification
    if files[FILE_SELECTED][len(files[FILE_SELECTED])-5:] == '.xlsx':
        DATA = pd.read_excel(files[FILE_SELECTED])
    if files[FILE_SELECTED][len(files[FILE_SELECTED])-4:] == '.csv':
        DATA = pd.read_csv(files[FILE_SELECTED])
        
    return DATA





def var_acquisition(DATA, COLUMN_SELECTED_IDX=-1, CHECK=True, interaction = True):
    """
    Selects and analyzes the target variable from a dataset.

    Parameters:
        DATA (pd.DataFrame): The dataset.
        COLUMN_SELECTED_IDX (int): Index of target column. If -1, asks the user.
        CHECK (bool): Whether to confirm target type with user.
        interaction (bool): Use interactive mode.

    Returns:
        DATA (pd.DataFrame): Dataset with cleaned target column.
        TARGET_COLUMN (str): Selected target column name.
        TARGET_TYPE (str): One of ['boolean', 'classes', 'continuous'].
        TARGET_DISTRIBUTION_TABLE (pd.DataFrame): Distribution of values in target.
    """    
    scewed_check = 0 #FLAG that of the scewed question in 3.2 -- by def is set to FALSE
    
    if interaction:
        # Menu implementation
        if COLUMN_SELECTED_IDX==-1:
            print(colored('\nPlease select the varaible to predict from the next list: ', 'green', attrs=['bold']))
            captions = []
            names = list(list(DATA.columns))  #cast column names into list to menu
            name = names[cutie.select(names, caption_indices=captions, selected_index=0)]
            res_list = [i for i, value in enumerate(names) if value == name]
            COLUMN_SELECTED_IDX = int(res_list[0])
    
    else:
        #Digit implementation
        if COLUMN_SELECTED_IDX==-1: #!!!!!!!!!Agregar a menu
            print('\nPlease select the varaible to predict from the next list: ')
            for col_idx,column in enumerate(DATA.columns):
                print('   '+str(col_idx)+') '+str(column))
            COLUMN_SELECTED_IDX = int(input("----> "))

    TARGET_COLUMN = DATA.columns[COLUMN_SELECTED_IDX] #the column name (is a string)
    print('Target column is: '+str(TARGET_COLUMN)+'\n')

    ### 1 drop rows from data by NaN in predict column
    idx_drop_from_predict = DATA.index[DATA[TARGET_COLUMN].isna() == True].tolist()
    DATA = DATA.drop(idx_drop_from_predict)
    # staticNaN = sf.Frame.from_pandas(DATA) #i use SF as imput of function due to funtion modifing the imput DF

    ### 3 chech predicted column type
    TARGET_TYPE = 'Not identify'
    unique_values = DATA[TARGET_COLUMN].unique()
    if len(unique_values) == 2: #if we have only two type of values it means is boolean
        TARGET_TYPE = 'boolean'
    
    ### 3.1 verify scewed predict data (to cast to boolean)
    SCEWED_TARGET_COL = pd.DataFrame(columns = ['Value id', 'Value', 'Count', 'Occurences (%)'])
    len_target_col = len(DATA[TARGET_COLUMN])
    for k,unique_val in enumerate(unique_values):
        unique_count = len(DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==unique_val]) 
        SCEWED_TARGET_COL.loc[len(SCEWED_TARGET_COL.index)] = [k, unique_val, unique_count, (unique_count/len_target_col)*100]
    drop_target_col_values = SCEWED_TARGET_COL[SCEWED_TARGET_COL['Occurences (%)']<10]['Value'] 
    target_col_values = SCEWED_TARGET_COL[SCEWED_TARGET_COL['Occurences (%)']>=10]['Value'] 
    
    ###In the case that values concentrate in only two values and we have something to drop <unque!=2>, ask if the user want to drop the columns that dont have much samples (making it a boolean) 
    if len(target_col_values) == 2 and len(unique_values)!=2:
        print(SCEWED_TARGET_COL) ### we could add a graph instead of a table ##
        
        if interaction:
            # Menu implementation
            scewed_check = cutie.prompt_yes_or_no("As the Table shows, the predicted column concentrate in only two values ("+str(target_col_values.values[0])+' and '+str(target_col_values.values[1])+"), do you want to only use this?")
            if scewed_check: #if TRUE we cast to boolean
                TARGET_TYPE = 'boolean'
                for drop_val in drop_target_col_values:
                    drop_row = DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==drop_val].index ####arrregalarrrrrrrrrrr!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    DATA = DATA.drop(drop_row)

        else:
            #Digit implementation
            print('As the Table shows, the predicted column concentrate in only two values ('+str(target_col_values[0])+' and '+str(target_col_values[1])+'), do you want to only use this? (1=YES/ 0=NO)')
            scewed_check = int(input("----> "))

        if interaction:
            # Menu implementation
            unbalance_check = cutie.prompt_yes_or_no("MAKE IT EVEN??? (unbalance target variable)")
            if unbalance_check: 
                min_entries = SCEWED_TARGET_COL.iloc[1,2]
                to_much_variable = SCEWED_TARGET_COL.iloc[0, 1]
                indexes_2_sort = DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==to_much_variable].index
                first = DATA.loc[np.random.choice(indexes_2_sort, size=min_entries, replace=False)]
                second = DATA[DATA[TARGET_COLUMN]==SCEWED_TARGET_COL.iloc[1, 1]]
                DATA = pd.concat([first, second])

                import seaborn as sns
                import matplotlib.pyplot as plt 
                plt.figure()
                sns.histplot(DATA, x=TARGET_COLUMN)
                plt.savefig('test.png')
                breakpoint()
                # for a in DATA[TARGET_COLUMN].unique():

                # for drop_val in drop_target_col_values:
                #     drop_row = DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==drop_val].index ####arrregalarrrrrrrrrrr!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #     DATA = DATA.drop(drop_row)
        
                
    ### 3.2 Coff of unique check     
    R = len(unique_values)/len_target_col #coeficient of number of unique in relation to number of rows
    number_of_classes = 10
    #if we have to many classes in relation of samples, ask to group the classes 
     #sceewed check is added becouse we dont want to over write what we done
    if R>(number_of_classes/100) and scewed_check!=1:    
        if len(unique_values)!=2: #dont enter if we already classify as boolean
            TARGET_TYPE = 'continuous'   
    if R<=(number_of_classes/100) and scewed_check!=1: 
        if len(unique_values)!=2: #dont enter if we already classify as boolean
            TARGET_TYPE = 'classes'      
            
    #if we find a match <!=Not identify>, we check if is it right (if check is TRUE)
    if TARGET_TYPE != 'Not identify' and CHECK: #take it depending if is boolean or classes

        print(colored('\nThe target column type is ', 'black', attrs=['bold']), end='')

        if TARGET_TYPE == 'continuous': #print depending of target typr
            # regression case
            print(colored(TARGET_TYPE, 'green', attrs=['bold']), end='')
            print(colored(', the elements are:', 'black', attrs=['bold']))

            for unique_element in DATA[TARGET_COLUMN].unique(): 
                print(colored(str(unique_element)+' ', 'green', attrs=['bold']), end='')

        
        else: 
            # classification case
            print(colored('discrete (', 'green', attrs=['bold']), end='')
            print(colored(TARGET_TYPE, 'green', attrs=['bold']), end='')
            print(colored(')', 'green', attrs=['bold']), end='')
            print(colored(', the classes are:', 'black', attrs=['bold']))
            print(colored(DATA[TARGET_COLUMN].unique(), 'green', attrs=['bold']))

        if interaction:
            # Menu implementation
            type_check = cutie.prompt_yes_or_no("Is this the type you want?")

        else:
            # Digit implementation
            print(colored('\n is this the type you want? (1=YES/ 0=NO)', 'black', attrs=['blink']))
            type_check = int(input("----> "))

        if type_check != 1:
            if TARGET_TYPE == 'continuous':

                if interaction:
                    # Menu implementation
                    cast_classes_check = cutie.prompt_yes_or_no("Do you want to convert the variable type to <classes> ?")
                else: 
                    # Digit implementation
                    print(colored('Do you want to convert the variable type to <classes> ? (1=YES/ 0=NO)', 'black', attrs=['blink']))
                    cast_classes_check = int(input("----> "))

                if cast_classes_check == 1:
                    print('Number of unique '+str(len(unique_values))+' of ' +str(len_target_col))
                    ###AUTORANGE #we already use ID and not the real value, we need to RECAST LATER
                    auto_range = np.linspace(min(unique_values),max(unique_values)+1,number_of_classes)
                    start_range_to_cast = auto_range[0]
                    for val_to_replace,end_range_to_cast in enumerate(auto_range[1:]):
                        DATA[TARGET_COLUMN] = DATA[TARGET_COLUMN].apply(lambda x: val_to_replace if x >= start_range_to_cast and x < end_range_to_cast else x)
                        start_range_to_cast = end_range_to_cast
                    TARGET_TYPE = 'classes'
                    # print('New unique count is '+str(len(DATA[TARGET_COLUMN].unique())))  
                    print('New range is:')
                    print(auto_range)

                    print(colored('\nThe target column type is ', 'black', attrs=['bold']), end='')
                    print(colored(TARGET_TYPE, 'green', attrs=['bold']), end='')
                    print(colored(', the classes are:', 'black', attrs=['bold']))

                    for unique_element in DATA[TARGET_COLUMN].unique(): 
                        print(colored(str(unique_element)+' ', 'green', attrs=['bold']), end='')   
                else:
                    TARGET_TYPE = 'Not identify'
    #if we couldnt find the target column type, we ask the user to define the ranges to clasifie
    if TARGET_TYPE == 'Not identify': # WORK ON IT OR DROP IT!!!!
        print('USER SELECT THE TARGET COLUMN TYPE')

    return DATA, TARGET_COLUMN, TARGET_TYPE, SCEWED_TARGET_COL




