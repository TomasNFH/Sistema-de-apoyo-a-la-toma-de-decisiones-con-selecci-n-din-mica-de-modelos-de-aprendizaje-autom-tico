import pandas as pd
import numpy as np

def ManualEDAfun(data):
    """
    Perform a manual exploratory data analysis (EDA) on a DataFrame.

    Parameters:
        data (pd.DataFrame): Input DataFrame to analyze.

    Returns:
        summary_by_column (pd.DataFrame): Summary of each column (distinct, missing, zero counts).
        summary_by_row (pd.DataFrame): Summary of each row with missing values.
    """
    N_row, N_col = data.shape

    summary_by_column = pd.DataFrame(columns = ['Column name', 'Distinct', 'Distinct (%)', 'Missing', 'Missing (%)', 'Zeros', 'Zeros (%)'])
    summary_by_row = pd.DataFrame(columns = ['Row id', 'Missing', 'Missing (%)'])
    
    for current_col in data.columns:
        current_data = data[current_col]

        unique_col = current_data.nunique()
        if unique_col!=0:
            Puniq_col = (unique_col/N_row)*100

        NaN_count_col = current_data.isna().sum()
        Pmiss_col = 0
        if NaN_count_col!=0:
            Pmiss_col = (NaN_count_col/N_row)*100

        zeros_col = (current_data == 0).sum()
        Pz_col = 0
        if zeros_col!=0:
            Pz_col = (zeros_col/N_row)*100

        summary_by_column.loc[len(summary_by_column.index)] = [current_col, unique_col, Puniq_col, NaN_count_col, Pmiss_col, zeros_col, Pz_col] 
   
    # Summary of missing data by row
    aux = np.asmatrix(data.isna().to_numpy()) 
    sum_aux = sum(np.transpose(aux)) 
    sum_aux = np.asarray(sum_aux)[0]
    matrix = np.matrix([data.index, sum_aux, (sum_aux/N_col)*100])
    summary_by_row = pd.DataFrame(np.transpose(matrix), columns=['Row id', 'Missing', 'Missing (%)'])

    return summary_by_column, summary_by_row
