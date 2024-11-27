from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#define exponential fuction to fit risk multiple
def monoExp(x, a, k, b):
    return a * np.exp(-k * x) + b

def init_prem_ibrd(file_path=Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data"), peril=None, year=None,want_plot=True):
    ibrd_bonds = pd.read_excel(file_path.joinpath('IBRD_bonds.xlsx'))
    if peril is not None: 
        flt_ibrd_bonds = ibrd_bonds[ibrd_bonds['Peril'] == peril]
        flt_ibrd_bonds = flt_ibrd_bonds.reset_index(drop=True)

    elif year is not None:
        flt_ibrd_bonds = ibrd_bonds[ibrd_bonds['Date'].isin(year)]
        flt_ibrd_bonds = flt_ibrd_bonds.reset_index(drop=True)

    else:
        flt_ibrd_bonds = ibrd_bonds.copy()
    #perform the fit
    params_prem_ibrd, cv = curve_fit(monoExp, flt_ibrd_bonds['Expected Loss'], flt_ibrd_bonds['Risk Multiple'])

    a, k, b = params_prem_ibrd
    x_fit = np.linspace(0.0, 10, 100)
    y_fitted = monoExp(x_fit, a, k, b)

    squaredDiffs = np.square(flt_ibrd_bonds['Risk Multiple'] - monoExp(flt_ibrd_bonds['Expected Loss'], a, k, b))
    squaredDiffsFromMean = np.square(flt_ibrd_bonds['Risk Multiple'] - np.mean(flt_ibrd_bonds['Risk Multiple']))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    #print(f"Risk multiple variance explained -> R² = {rSquared}")

    if want_plot:
        plt_data = flt_ibrd_bonds[['Expected Loss', 'Risk Multiple', 'Date', 'Size']]
        palette = { '2017': 'tab:red',
                    '2018': 'tab:orange',
                    '2019': 'tab:pink',
                    '2020': 'tab:purple',
                    '2021': 'tab:blue',
                    '2023': 'tab:green',
                    '2024': 'tab:olive'
                    }
        sns.color_palette("rocket")
        sns.scatterplot(data=plt_data, x='Expected Loss', y='Risk Multiple', hue='Date', size="Size",
                    sizes=(40, 400), alpha=.5)
        plt.plot(x_fit, y_fitted, color = 'orange', label="fitted")
        plt.xlabel('Expected Loss [%]')
        plt.ylabel('Risk Multiple')
        plt.show()

    return params_prem_ibrd