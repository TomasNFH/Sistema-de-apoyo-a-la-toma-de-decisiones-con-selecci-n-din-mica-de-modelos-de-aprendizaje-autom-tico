import eda
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from termcolor import colored


def data_cleaning(data, min_porcentage_col = 10, min_porcentage_row = 0):

    manualEDA, missing4rows = eda.ManualEDAfun(data)
    # drop ror & col by missing (%)
    drop_col = manualEDA[manualEDA['Missing (%)']>min_porcentage_col]['Column name'] 
    drop_row = missing4rows[missing4rows['Missing (%)']>min_porcentage_row]['Row id']
    print(colored('\nResultado de Data Cleaning:', 'red', attrs=['bold']))
    print(colored('  Cantidad de columnas eliminadas: '+
          str(len(drop_col))+
          ' of '+
          str(len(data.columns))+
          ' ('+
          str( round(len(drop_col)/len(data.columns)*100, 2) )+
          '%)', 'red'))
    print(colored('  Cantidad de filas eliminadas: '+
          str(len(drop_row))+
          ' of '+
          str(len(data))+
          ' ('+
          str( round(len(drop_row)/len(data)*100, 2) )+
          '%)', 'red'))
    # print(data.olumns[drop_col])
    data = data.drop(drop_col, axis=1)
    data = data.drop(drop_row)

    return data, drop_col, drop_row

def data_normF(data, FLAG=1):
    
    data_norm = data
    ### BYPASS NORMALIZATION ###
    if FLAG == 0:
        data_norm = data
    else:
        ### min-max normalization ###
        if FLAG:  
            scaler = MinMaxScaler()
            scaler.fit(data)
            X_norm_arr = scaler.transform(data)
        ### Z-score normalization ###
        if FLAG == 2:
            scaler = StandardScaler()
            scaler.fit(data)
            X_norm_arr = scaler.transform(data) 
        #remove not norm column and add a norm one
        for idx,col in enumerate(data.columns):
            data_norm = data_norm.drop(col, axis=1)
            data_norm.insert(len(data.columns)-1, col, X_norm_arr[:,idx], True)
    
    return data_norm

