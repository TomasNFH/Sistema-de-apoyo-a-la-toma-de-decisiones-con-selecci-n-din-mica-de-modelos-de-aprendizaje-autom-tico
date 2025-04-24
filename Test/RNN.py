import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

breakpoint()
DATA = pd.read_csv('/Users/tomasferraz/Sistema-de-apoyo-a-la-toma-de-decisiones-con-selecci-n-din-mica-de-modelos-de-aprendizaje-autom-tico/input/HC_nefro/OUT/HDwithCM.csv')
extended = pd.read_csv('/Users/tomasferraz/Sistema-de-apoyo-a-la-toma-de-decisiones-con-selecci-n-din-mica-de-modelos-de-aprendizaje-autom-tico/input/HC_nefro/OUT/HDextended.csv', sep=',') 

pacient_fst_month_data = pd.DataFrame()

for id in DATA['HASH'].dropna().unique():
    pacient = DATA[DATA['HASH']==id]
    pacient = pacient.sort_values(by='MES', ascending=True)
    print(id)
    first_month = pacient.iloc[0,0]

    if first_month != 1: 
        # hashes_pacient_starters.append(id)
        # month.append(first_month)
        cr_data = pacient[pacient['MES']==first_month]
        pacient_fst_month_data = pd.concat([pacient_fst_month_data, cr_data])
        # breakpoint()

breakpoint()

