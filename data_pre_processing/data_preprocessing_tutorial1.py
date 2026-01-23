#### data pre-processing for ML using Python #################
#### tutorial on how to use pandas library ###################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#### read a .csv file and save it as a dataframe variable #############
data_frame  =   pd.read_csv("diabetes.csv")
print(data_frame)

#### data preprocessing ###########################################

#### inspect how the data for missing values ######################
# print(data_frame.info()) # df.info() - gives a summary of the dataset that has been read
# print(data_frame.isnull().sum()) # df.isnull().sum() - gives the number of null datapoints in each coloumn
# print(data_frame.isnull())

#### statistical summary and visualizing outliers #################
# print(data_frame.describe())

### the box plot visualization uses the concept of quartiles
fig, axs    =   plt.subplots(len(data_frame.columns), 1, figsize=(7,18), dpi=95)
for i, col in enumerate(data_frame.columns):
    axs[i].boxplot(data_frame[col], vert=False)
    axs[i].set_ylabel(col)
plt.tight_layout()

### there are three values of quartiles: lower quartile, median and upper quartile
### 25% of data is below lower quartile, 50 % is below median and 75% is below upper quartile
### the Interquartile Range (IQR) is used to define and filter outiler data points
#### following lines of code uses IQR parameter to filter outliers

# say one wants to filter the data only using one parameter
parameter1                      =       "DiabetesPedigreeFunction"
q1, q3                          =       np.percentile(data_frame[parameter1], [25,75])
print(q1,q3)
iqr                             =       q3-q1
whisker_val                     =       1.0
lower_datum                     =       q1 - (whisker_val*iqr)
upper_datum                     =       q3 + (whisker_val*iqr)
print(lower_datum, upper_datum)
outliers_discard_filter         =       (lower_datum<=data_frame[parameter1]) & (data_frame[parameter1]<=upper_datum)
filtered_df                     =       data_frame[outliers_discard_filter]
plt.figure(2, figsize=(7,2))
plt.boxplot(filtered_df[parameter1], vert=False)
plt.ylabel(parameter1)
plt.tight_layout()
plt.show()

### there are three values of quartiles: lower quartile, median and upper quartile
### 25% of data is below lower quartile, 50 % is below median and 75% is below upper quartile
### the Interquartile Range (IQR) is used to define and filter outiler data points
#### preceding lines of code uses IQR parameter to filter outliers