import pandas as pd
import numpy as np
from typing import Sequence
import matplotlib.pyplot as plt

############### Auxilary functions ###############################################
def quartile_data_cleaner(
        file_name,
        parameters: str | Sequence[str],
        whisker: float | Sequence[float]):
    """
    author      :   Jwalit Panchal
    use         :   remove the outliers from various parameters of a dataset using tolerence on quartiles
    parameters  :   file_name- str, name of the file to read, mostly csv
                    parameters- str, list of strings, the parameters for which data needs to be cleaned
                    whisker- float, list of floats, the multiplies to iqr used for marking lower and upper data pts
    """
    data_frame      =       pd.read_csv(file_name)

    if (type(whisker)==float):
        q1, q3              =       np.percentile(data_frame[parameters], [25,75])
        iqr                 =       q3-q1
        lower_datum         =       q1-(whisker*iqr)
        upper_datum         =       q3+(whisker*iqr)
        outlier_filter      =       (data_frame[parameters]>=lower_datum) & (data_frame[parameters]<=upper_datum)
        data_frame          =       data_frame[outlier_filter]
    else:
        for i, param in enumerate(parameters):
            q1, q3              =       np.percentile(data_frame[param], [25,75])
            iqr                 =       q3-q1
            lower_datum         =       q1-(whisker[i]*iqr)
            upper_datum         =       q3+(whisker[i]*iqr)
            outlier_filter      =       (data_frame[param]>=lower_datum) & (data_frame[param]<=upper_datum)
            data_frame          =       data_frame[outlier_filter]
    
    return data_frame

def quartile_data_cleaner_plotter(
        file_name,
        parameters: str | Sequence[str],
        whisker: float | Sequence[float]):
    """
    author      :   Jwalit Panchal
    use         :   remove the outliers from various parameters of a dataset using tolerence on quartiles and plot the raw and filtered data
    parameters  :   file_name- str, name of the file to read, mostly csv
                    parameters- str, list of strings, the parameters for which data needs to be cleaned
                    whisker- float, list of floats, the multiplies to iqr used for marking lower and upper data pts
    """  
    data_frame      =       pd.read_csv(file_name)
    if (type(whisker)==float):
        fig, axs        =       plt.subplots(ncols=2, figsize=(7,2), sharey=True)
        q1, q3              =       np.percentile(data_frame[parameters], [25,75])
        iqr                 =       q3-q1
        lower_datum         =       q1-(whisker*iqr)
        upper_datum         =       q3+(whisker*iqr)
        axs[0].boxplot(data_frame[parameters], vert=False, whis=whisker)
        axs[0].set_xlabel(parameters)
        axs[0].xaxis.set_label_coords(0.5, 0.1)
        axs[0].tick_params(direction="in")
        outlier_filter      =       (data_frame[parameters]>=lower_datum) & (data_frame[parameters]<=upper_datum)
        data_frame          =       data_frame[outlier_filter]
        axs[1].boxplot(data_frame[parameters], vert=False, whis=whisker)
        axs[1].set_xlabel(parameters)
        axs[1].xaxis.set_label_coords(0.5, 0.1)
        axs[1].tick_params(direction="in")
    else:
        fig, axs        =       plt.subplots(len(whisker),2, figsize=(7, int(2*len(whisker))), sharey=True)
        for i, param in enumerate(parameters):
            q1, q3              =       np.percentile(data_frame[param], [25,75])
            iqr                 =       q3-q1
            lower_datum         =       q1-(whisker[i]*iqr)
            upper_datum         =       q3+(whisker[i]*iqr)
            axs[i,0].boxplot(data_frame[param], vert=False, whis=whisker[i])
            axs[i,0].set_xlabel(param)
            axs[i,0].xaxis.set_label_coords(0.5, 0.1)
            axs[i,0].tick_params(direction="in")
            outlier_filter      =       (data_frame[param]>=lower_datum) & (data_frame[param]<=upper_datum)
            data_frame          =       data_frame[outlier_filter]
            axs[i,1].boxplot(data_frame[param], vert=False, whis=whisker[i])
            axs[i,1].set_xlabel(param)
            axs[i,1].xaxis.set_label_coords(0.5, 0.1)
            axs[i,1].tick_params(direction="in")

    plt.tight_layout()
    plt.show()
    return

def quartile_full_data_cleaner(
        file_name: str,
        whisker: float | Sequence[float]
        ):
    """
    author      :   Jwalit Panchal
    use         :   remove the outliers from all parameters of a dataset using tolerence on quartiles and plot the raw and filtered data
    parameters  :   file_name- str, name of the file to read, mostly csv
                    whisker- float, list of floats, the multiplies to iqr used for marking lower and upper data pts
                    if whisker is given as a list of floats, then the size should match the number of coloums present
                    in the dataframe and the corresponding whisker value whisker[i] will be used screen the i-th col
    """
    data_frame              =       pd.read_csv(file_name)
    if (type(whisker)==float):
        for i, col in enumerate(data_frame.columns):
            q1,q3           =       np.percentile(data_frame[col], [25,75])
            iqr             =       q3-q1
            lower_datum     =       q1-(iqr*whisker)
            upper_datum     =       q3+(iqr*whisker)
            outlier_filter  =       (data_frame[col]>=lower_datum) & (data_frame[col]<=upper_datum)
            data_frame      =       data_frame[outlier_filter]
    else:
        for i, col in enumerate(data_frame.columns):
            q1,q3           =       np.percentile(data_frame[col], [25,75])
            iqr             =       q3-q1
            lower_datum     =       q1-(iqr*whisker[i])
            upper_datum     =       q3+(iqr*whisker[i])
            outlier_filter  =       (data_frame[col]>=lower_datum) & (data_frame[col]<=upper_datum)
            data_frame      =       data_frame[outlier_filter]
    
    return data_frame

############### Auxilary functions ###############################################

##### eg. code for testing Aux. fns. #############################################
print("Eg.1\n")
file_name           =               "diabetes.csv"
parameter           =               "Pregnancies"
whisker             =               1.0
eg_df               =               pd.read_csv(file_name)
filtered_df         =               quartile_data_cleaner(file_name=file_name, parameters=parameter, whisker=whisker)

plt.figure(1, figsize=(7,2))
plt.boxplot(eg_df[parameter], vert=False, whis=whisker)
plt.ylabel(parameter)
plt.tight_layout()

plt.figure(2, figsize=(7,2))
plt.boxplot(filtered_df[parameter], vert=False, whis=whisker)
plt.ylabel(parameter)
plt.tight_layout()

plt.show()

print("Eg.2\n")
file_name           =               "diabetes.csv"
parameter           =               ["Age", "Glucose", "Pregnancies"]
whisker             =               [1.0,1.0,1.0]
quartile_data_cleaner_plotter(file_name=file_name, parameters=parameter, whisker=whisker)

print("Eg.3\n")
file_name           =               "diabetes.csv"
whisker             =               1.0
print(eg_df,"\n")
full_cleaned_df     =           quartile_full_data_cleaner(file_name=file_name, whisker=whisker)
print(full_cleaned_df)
print(eg_df.info(),"\t",full_cleaned_df.info())
##### eg. code for testing Aux. fns. #############################################