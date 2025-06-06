from System.Modules import eda
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from termcolor import colored
import pandas as pd


def data_cleaning(data, min_porcentage_col = 10, min_porcentage_row = 0):
    """
    Cleans a DataFrame by removing columns and rows based on missing value percentage thresholds.

    Parameters:
        data (pd.DataFrame): Input dataset.
        min_percentage_col (float): Max allowed percentage of missing values per column.
        min_percentage_row (float): Max allowed percentage of missing values per row.

    Returns:
        cleaned_data (pd.DataFrame): Cleaned dataset.
        dropped_columns (pd.Series): Names of dropped columns.
        dropped_rows (pd.Series): Indexes of dropped rows.
    """
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
    """
    Normalizes a DataFrame using Min-Max or Z-score normalization.

    Parameters:
        data (pd.DataFrame): Input data to normalize.
        method (int): Normalization method:
                      0 - No normalization,
                      1 - Min-Max scaling,
                      2 - Z-score standardization.

    Returns:
        normalized_data (np.ndarray): Normalized data as numpy array.
    """   
    # breakpoint()
    # data_norm = data
    ### BYPASS NORMALIZATION ###
    if FLAG == 0:
        data_norm = data
    else:
        ### min-max normalization ###
        if FLAG == 1:  
            scaler = MinMaxScaler()
            scaler.fit(data)
            data_norm = scaler.transform(data)
        ### Z-score normalization ###
        if FLAG == 2:
            scaler = StandardScaler()
            scaler.fit(data)
            data_norm = scaler.transform(data) 
        # #remove not norm column and add a norm one
        # for idx,col in enumerate(data.columns):
        #     data_norm = data_norm.drop(col, axis=1)
        #     data_norm.insert(len(data.columns)-1, col, X_norm_arr[:,idx], True)
        
    



    # if method == 0:
    #     return data

    # scaler = MinMaxScaler() if method == 1 else StandardScaler()
    # normalized_data = scaler.fit_transform(data)


    return data_norm

