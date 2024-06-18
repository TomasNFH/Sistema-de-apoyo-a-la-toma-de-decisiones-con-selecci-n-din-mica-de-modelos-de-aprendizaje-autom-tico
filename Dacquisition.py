import os
import pandas as pd
import auxiliary_fun
import static_frame as sf
from termcolor import colored
import numpy as np
import cutie



def Data_adq(file_selected_idx = -1):
    files = []
    names = [] #just the name to print
    for dirname, _, filenames in os.walk(r'C:\Users\tomas\Codigos\Pasantia\VScodeVER\Input'):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))    
            names.append(filename) 

    #Digit implementation
    # if file_selected_idx == -1: #if ==-1 ask for the idx
    #     print(colored('The possible files to load are: \n', 'black', attrs=['bold']))

    #     for file_idx,name in enumerate(names):
    #           print('----> File number '+str(file_idx)+': '+str(name)) 
    #     file_selected_idx = int(input("\n\nDigit the file number: "))
    # print('File selected is: '+str(files[file_selected_idx]))
    
    #Menu implementation
    print(colored('The possible files to load are: \n', 'green', attrs=['bold']))
    captions = []
    name = names[cutie.select(names, caption_indices=captions, selected_index=0)]
    res_list = [i for i, value in enumerate(names) if value == name]
    print(int(res_list[0]))
    file_selected_idx = int(res_list[0])
    
    #data type verification
    if files[file_selected_idx][len(files[file_selected_idx])-5:] == '.xlsx':
        data = pd.read_excel(files[file_selected_idx])
    if files[file_selected_idx][len(files[file_selected_idx])-4:] == '.csv':
        data = pd.read_csv(files[file_selected_idx])
        
    return data



def variable_adq(data, column_selected_idx=-1, check=True):
    ret_data = data #is the data entry cleaned (NaN drop, cast) --- by def return the imput
    scewed_check = 0 #FLAG that of the scewed question in 3.2 -- by def is set to FALSE
    
    #Digit implementation
    # if column_selected_idx==-1: #!!!!!!!!!Agregar a menu
    #     print('\nPlease select the varaible to predict from the next list: ')
    #     for col_idx,column in enumerate(data.columns):
    #         print('   '+str(col_idx)+') '+str(column))
    #     column_selected_idx = int(input("----> "))


    #Menu implementation
    if column_selected_idx==-1:
        print(colored('\nPlease select the varaible to predict from the next list: ', 'green', attrs=['bold']))
        captions = []
        names = list(list(data.columns))  #cast column names into list to menu
        name = names[cutie.select(names, caption_indices=captions, selected_index=0)]
        res_list = [i for i, value in enumerate(names) if value == name]
        # print(int(res_list[0]))
        column_selected_idx = int(res_list[0])

    predicted_column = data.columns[column_selected_idx] #the column name (is a string)
    print('Target column is: '+str(predicted_column)+'\n')

    ### 1 drop rows from data by NaN in predict column
    idx_drop_from_predict = data.index[data[predicted_column].isna() == True].tolist()
    dataNaN = data.drop(idx_drop_from_predict)
    staticNaN = sf.Frame.from_pandas(dataNaN) #i use SF as imput of function due to funtion modifing the imput DF
    
    ### 2 cast ONLY predicted column to int (do this to work internally, in the result we DE-CAST)
    dataCAST, uniqueVALRRR, id_uniqueRRR = auxiliary_fun.unique_to_int(staticNaN, predicted_column)
#     staticCAST = sf.Frame.from_pandas(dataCAST) 
#     dataREcast = unique_to_int_reverse(staticCAST, predicted_column, uniqueVALRRR, id_uniqueRRR)
    
    ### 3 chech predicted column type
    target_type = 'Not identify'
    unique_values = dataNaN[predicted_column].unique()
    if len(unique_values) == 2: #if we have only two type of values it means is boolean
        target_type = 'boolean'
    
    ### 3.1 verify scewed predict data (to cast to boolean)
    scewed_target_col = pd.DataFrame(columns = ['Value id', 'Value', 'Count', 'Occurences (%)'])
    len_target_col = len(dataNaN[predicted_column])
    for k,unique_val in enumerate(unique_values):
        unique_count = len(dataNaN[predicted_column][dataNaN[predicted_column]==unique_val]) 
        scewed_target_col.loc[len(scewed_target_col.index)] = [k, unique_val, unique_count, (unique_count/len_target_col)*100]
    drop_target_col_values = scewed_target_col[scewed_target_col['Occurences (%)']<10]['Value'] 
    target_col_values = scewed_target_col[scewed_target_col['Occurences (%)']>=10]['Value'] 
    
    ###In the case that values concentrate in only two values and we have something to drop <unque!=2>, ask if the user want to drop the columns that dont have much samples (making it a boolean) 
    if len(target_col_values) == 2 and len(unique_values)!=2:
        print(scewed_target_col) ### we could add a graph instead of a table ##

        #def
        # print('As the Table shows, the predicted column concentrate in only two values ('+str(target_col_values[0])+' and '+str(target_col_values[1])+'), do you want to only use this? (1=YES/ 0=NO)')
        # scewed_check = int(input("----> "))

        #menu
        scewed_check = cutie.prompt_yes_or_no("As the Table shows, the predicted column concentrate in only two values ("+str(target_col_values[0])+' and '+str(target_col_values[1])+"), do you want to only use this?")

        if scewed_check: #if TRUE we cast to boolean
            target_type = 'boolean'
            for drop_val in drop_target_col_values:
                drop_row = dataNaN['Tipo_salida'][dataNaN['Tipo_salida']==drop_val].index
                dataNaN = dataNaN.drop(drop_row)
                
    ### 3.2 Coff of unique check     
    R = len(unique_values)/len_target_col #coeficient of number of unique in relation to number of rows
    number_of_classes = 10
    
    if R>(number_of_classes/100) and scewed_check!=1: #if we have to many classes in relation of samples, ask to group the classes    
                          #sceewed check is added becouse we dont want to over write what we done
        if len(unique_values)!=2: #dont enter if we already classify as boolean
            target_type = 'continuous'   
    if R<=(number_of_classes/100) and scewed_check!=1: 
        if len(unique_values)!=2: #dont enter if we already classify as boolean
            target_type = 'classes'      
            
    #if we find a match <!=Not identify>, we check if is it right (if check is TRUE)
    if target_type != 'Not identify' and check: #ake it depending if is boolean or classes
        
        #print depending of target typr
        if target_type == 'continuous':
#             print('\nThe target column type is ', end='')
# +target_type+', the elements are:')
            print(colored('\nThe target column type is ', 'black', attrs=['bold']), end='')
            print(colored(target_type, 'green', attrs=['bold']), end='')
            print(colored(', the elements are:', 'black', attrs=['bold']))
        else: #discrete case
            print(colored('\nThe target column type is ', 'black', attrs=['bold']), end='')
            print(colored('discrete (', 'green', attrs=['bold']), end='')
            print(colored(target_type, 'green', attrs=['bold']), end='')
            print(colored(')', 'green', attrs=['bold']), end='')
            print(colored(', the classes are:', 'black', attrs=['bold']))


        if target_type == 'boolean': print(colored(dataNaN[predicted_column].unique(), 'green', attrs=['bold']))
        for unique_element in dataNaN[predicted_column].unique(): 
            print(colored(str(unique_element)+' ', 'green', attrs=['bold']), end='')
            
#         print(dataNaN[predicted_column].unique()) #return classes before cast <dataNaN>
        
#         print('\n is this the type you want? (1=YES/ 0=NO)')

        #def
        # print(colored('\n is this the type you want? (1=YES/ 0=NO)', 'black', attrs=['blink']))
        # type_check = int(input("----> "))

        #Menu imp
        type_check = cutie.prompt_yes_or_no("Is this the type you want?")
        if type_check != 1:
            if target_type == 'continuous':

                #def
                print(colored('Do you want to convert the variable type to <classes> ? (1=YES/ 0=NO)', 'black', attrs=['blink']))
                cast_classes_check = int(input("----> "))
                
                #menu
                cast_classes_check = cutie.prompt_yes_or_no("Do you want to convert the variable type to <classes> ?")

                if cast_classes_check == 1:
                    # print('Enter autorange (R is '+str(R)+').')
                    print('Number of unique '+str(len(unique_values))+' of ' +str(len_target_col))

                    ###AUTORANGE #we already use ID and not the real value, we need to RECAST LATER
                    auto_range = np.linspace(min(unique_values),max(unique_values)+1,number_of_classes)
                    start_range_to_cast = auto_range[0]
                    for val_to_replace,end_range_to_cast in enumerate(auto_range[1:]):
                        dataNaN[predicted_column] = dataNaN[predicted_column].apply(lambda x: val_to_replace if x >= start_range_to_cast and x < end_range_to_cast else x)
                        start_range_to_cast = end_range_to_cast
                    target_type = 'classes'
                    # print('New unique count is '+str(len(dataNaN[predicted_column].unique())))  
                    print('New range is:')
                    print(auto_range)

                    print(colored('\nThe target column type is ', 'black', attrs=['bold']), end='')
                    print(colored(target_type, 'green', attrs=['bold']), end='')
                    print(colored(', the classes are:', 'black', attrs=['bold']))

                    for unique_element in dataNaN[predicted_column].unique(): 
                        print(colored(str(unique_element)+' ', 'green', attrs=['bold']), end='')   

                else:
                    target_type = 'Not identify'


    #if we couldnt find the target column type, we ask the user to define the ranges to clasifie
    if target_type == 'Not identify': #work on it
        print('USER SELECT THE TARGET COLUMN TYPE')
        
       
    #q pasa si dev cast? 
    #dentro de la funcion dataCAst no lo uso
    
#   ret_data = dataCAST #en algun momento lo casteo mas adelante, sacar y usar este cast
    ret_data = dataNaN
    
#     print('dataNAN')
#     print(dataNaN)
#     print('dataC')
#     print(dataCAST)
    return ret_data, predicted_column, target_type, uniqueVALRRR, id_uniqueRRR, scewed_target_col