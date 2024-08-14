import pandas as pd


def ManualEDAfun(data):

    manualEDA = pd.DataFrame(columns = ['Column name', 'Distinct', 'Distinct (%)', 'Missing', 'Missing (%)', 'Zeros', 'Zeros (%)'])
    missing4rows = pd.DataFrame(columns = ['Row id', 'Missing', 'Missing (%)'])
    N_row, N_col = data.shape
    for current_col in data.columns:
        current_data = data[current_col]
        #Distinct
        unique_col = current_data.nunique()
        if unique_col!=0:
            Puniq_col = (unique_col/N_row)*100
        #Missing
        NaN_count_col = current_data.isna().sum()
        Pmiss_col = 0
        if NaN_count_col!=0:
            Pmiss_col = (NaN_count_col/N_row)*100
        #Zeros
        zeros_col = (current_data == 0).sum()
        Pz_col = 0
        if zeros_col!=0:
            Pmz_col = (zeros_col/N_row)*100
        manualEDA.loc[len(manualEDA.index)] = [current_col, unique_col, Puniq_col, NaN_count_col, Pmiss_col, zeros_col, Pz_col] 
    # missing values by row
    # breakpoint()
    for row in data.iterrows():
        NaN_count_row = row[1].isna().sum()
        missing4rows.loc[len(missing4rows.index)] = [row[0], NaN_count_row, (NaN_count_row/N_col)*100] 
        
    return manualEDA, missing4rows