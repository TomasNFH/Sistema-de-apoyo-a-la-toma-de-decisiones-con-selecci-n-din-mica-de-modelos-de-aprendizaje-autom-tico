import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from VikParuchuri import functions
from sklearn.preprocessing import StandardScaler

np.random.seed(0)


os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir)) #go back to Test folder
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir)) ##go back to main folder

breakpoint()
DATA = pd.read_csv('Input/Nefrología/HDwithCM.csv')
extended = pd.read_csv('Input/Nefrología/HDextended.csv')
# DATA = pd.read_csv('/Users/tomasferraz/Sistema-de-apoyo-a-la-toma-de-decisiones-con-selecci-n-din-mica-de-modelos-de-aprendizaje-autom-tico/input/HC_nefro/OUT/HDwithCM.csv')
# extended = pd.read_csv('/Users/tomasferraz/Sistema-de-apoyo-a-la-toma-de-decisiones-con-selecci-n-din-mica-de-modelos-de-aprendizaje-autom-tico/input/HC_nefro/OUT/HDextended.csv', sep=',') 

breakpoint()

# Define predictors and target
PREDICTORS = ["PADPRED", "HDFAVNATIVA", "V39"]
TARGET = "CAUSA_BAJA"

# Scale our data to have mean 0
scaler = StandardScaler()
DATA[PREDICTORS] = scaler.fit_transform(DATA[PREDICTORS])


# pacient_fst_month_data = pd.DataFrame()
# pacient_scnd_month_data = pd.DataFrame()
# pacient_thrd_month_data = pd.DataFrame()

stoppp = False
for id in DATA['HASH'].dropna().unique():
    if not stoppp:
        pacient = DATA[DATA['HASH']==id]
        pacient = pacient.sort_values(by='MES', ascending=True)
        first_month = pacient.iloc[0,0]
        
        if first_month != 1: 
            breakpoint()

            pacient = pacient.drop(6458)
            data = pacient[PREDICTORS]

            # cr_data = pacient[pacient['MES']==first_month]
            # pacient_fst_month_data = pd.concat([pacient_fst_month_data, cr_data])

            # cr_data = pacient[pacient['MES']==first_month+1]
            # pacient_scnd_month_data = pd.concat([pacient_scnd_month_data, cr_data])

            # cr_data = pacient[pacient['MES']==first_month+2]
            # pacient_thrd_month_data = pd.concat([pacient_thrd_month_data, cr_data])

            # stoppp = True
            # # breakpoint()

#data is the data that i will use to train the model (RNN for a unique pacient)
breakpoint()




breakpoint()

