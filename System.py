import auxiliary_fun
import Dacquisition
import DprepNcleaning
import eda
import Mbuilding


def SystemFUN(file_selected=-1, column_selected=-1):
    
    # ### Step 1 ###
    dataADQ = Dacquisition.Data_adq(file_selected)
    data, predicted_column, target_type, unique4re_cast, id_unique, scewed_target_col = Dacquisition.variable_adq(dataADQ, column_selected, check=True) #tema de dataNaN y dataCLEAN juntar

    ### Step 2 ###    
#     data, uniqueVAL, id_unique = unique_to_int(data, 'Sexo') # hacer auto (POR AHORA PASARLO AL MAIN) esto o q la base de datos  ya lo traiga incorporado
                                                                #tipo requirements que se hacen en el main antes de entrar a SYSTEM
    print('\n')
    print('ANTES')
    print(data.head())
    #AUTO-CAST strings to int (if <unique == 1> we save the value as a key and drop the column for the model)
    for column in data:
        print('\n')
        print(column)
#         print(data[column])
#         print(len(data[column].unique()))
        
        #if there is only one unique, we save the KEY and drop the column
        if len(data[column].unique()) == 1:
            print('UNIQUE DROP')
            unique_val = data[column][0] #preparar para recuperar para el final del modelado (guardar en un df)
            data = data.drop(column, axis=1)
        else: #if we drop we cant acces the colum type    
            column_type = data[column].dtype
            #if the column contain string we cast it to int
            if column_type == 'object':
                 data, uniqueVAL, id_unique = auxiliary_fun.unique_to_int(data, column) #guardar todo esto en un DF para recuperar
                
#     data, uniqueVAL, id_unique = unique_to_int(data, 'Domain')
#     data, uniqueVAL, id_unique = unique_to_int(data, 'Domain Code')
#     data, uniqueVAL, id_unique = unique_to_int(data, 'Area')
#     data, uniqueVAL, id_unique = unique_to_int(data, 'Element')
#     data, uniqueVAL, id_unique = unique_to_int(data, 'Item')
#     data, uniqueVAL, id_unique = unique_to_int(data, 'Unit')
#     data, uniqueVAL, id_unique = unique_to_int(data, 'Flag')
#     data, uniqueVAL, id_unique = unique_to_int(data, 'Flag Description')
    
#    Flag Flag Description


    print('\n')
    print('DESPUES')
    print(data.head())

    data = DprepNcleaning.data_cleaning(data, min_porcentage_col = 10, min_porcentage_row = 0)
        ### Step 3: Exploratory Data Analyzis ###
    #Manual
    manualEDA, missing4rows = eda.ManualEDAfun(data) #data no tiene a predicted column

        ### Step 5: Model Building       
#     model_info = model_shake(data, predicted_column, target_type)
    model_info = Mbuilding.model_shake(DATA=data, PREDICTED_CL=predicted_column, TARGET_TY=target_type)
    
    
    return model_info
# variable_adq(data, column_selected_idx=-1, check=False):
#     return ret_data, predicted_column, target_type, uniqueVALRRR, scewed_target_col


