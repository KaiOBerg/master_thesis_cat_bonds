from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ibrd_bonds = pd.read_excel(r"C:\Users\kaibe\Documents\ETH_Zurich\Thesis\Data\IBRD_bonds.xlsx")

#define exponential fuction to fit risk multiple
def monoExp(x, a, k, b):
    return a * np.exp(-k * x) + b

def init_prem_ibrd():

    #perform the fit
    params_prem_ibrd, cv = curve_fit(monoExp, ibrd_bonds['Expected Loss'], ibrd_bonds['Risk Multiple'])

    a, k, b = params_prem_ibrd
    x_fit = np.linspace(0.5, ibrd_bonds['Expected Loss'].max(), 100)
    y_fitted = monoExp(x_fit, a, k, b)

    squaredDiffs = np.square(ibrd_bonds['Risk Multiple'] - monoExp(ibrd_bonds['Expected Loss'], a, k, b))
    squaredDiffsFromMean = np.square(ibrd_bonds['Risk Multiple'] - np.mean(ibrd_bonds['Risk Multiple']))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    #print(f"Risk multiple variance explained -> R² = {rSquared}")


    plt.scatter(ibrd_bonds['Expected Loss'], ibrd_bonds['Risk Multiple'])
    #Annotate each point with its label
    for i in range(len(ibrd_bonds['Risk Multiple'])):
        plt.annotate(
            ibrd_bonds['Date'][i],  #Text to display
            (ibrd_bonds['Expected Loss'][i], ibrd_bonds['Risk Multiple'][i]),  #Point location
            xytext=(20, 20),  #Offset position (adjust as needed)
            textcoords='offset points',  #Use offset in points
            arrowprops=dict(arrowstyle='-', color='black', lw=0.5),  #Arrow style
            ha='center',  #Horizontal alignment
            va='bottom'   #Vertical alignment
        )
    plt.plot(x_fit, y_fitted, color = 'orange', label="fitted")
    plt.xlabel('Expected Loss [%]')
    plt.ylabel('Risk Multiple')
    plt.show()

    return params_prem_ibrd