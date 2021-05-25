# Importing matplotlib modules
import matplotlib.pyplot as plt

# Importing pandas modules
import pandas as pd

# Importing numpy modules
import numpy as np

# Importing seaborn modules
import seaborn as sns

def analyze_variable(data, var):
    # Create grid for figures
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    
    # Plot 
    sns.histplot(data=data[var], kde=True, ax=axs[0][0], stat='density')
    axs[0][0].set_title('Distribución')
    
    sns.boxplot(data=data[var], ax=axs[0][1])
    axs[0][1].set_title('Boxplot')
    
    sns.histplot(data=data[var][data['Outcome'] == 0], kde=True, ax=axs[1][0], stat='density')
    axs[1][0].set_title('Distribución si no posee diabetes')
    
    sns.histplot(data=data[var][data['Outcome'] == 1], kde=True, ax=axs[1][1], stat='density')
    axs[1][1].set_title('Distribución si posee diabetes')
    
    axs[1][1].set_ylim(axs[1][0].get_ylim())
    
    # Show
    plt.show()
    
def get_outliers(data, var):
    # Usa criterio de "Outlier Leve"
    # extraído de https://es.wikipedia.org/wiki/Valor_at%C3%ADpico 
    q1 = data[var].quantile(0.25)
    q3 = data[var].quantile(0.75)
    iqr = q3 - q1
    mean = data[var].mean()
    ret = []
    for value in data[var]:
        if value < (q1 - 1.5 * iqr) or value > (q3 + 1.5 * iqr):
            ret.append(value)
    return ret

def remove_outliers(data, var): 
    outliers = get_outliers(data, var)
    for outlier in outliers:
        data[var].replace(outlier, np.nan, inplace=True)
        