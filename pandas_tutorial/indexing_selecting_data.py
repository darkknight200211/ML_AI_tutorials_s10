import pandas as pd

#### read data from a .csv file
df      =       pd.read_csv("/home/jwalitnpanchal/college_work/sem10/ML_AI/data_pre_processing/diabetes.csv")
# print(df)

####### The [] operator uses ####################################
# selecting one coloumn from the dataframe
one_col =       df["Glucose"]
# print(one_col)

# selecting multiple cols from the dataframe
mult_cols   =   df[["Pregnancies", "BMI", "Age"]]
# print(mult_cols)
####### The [] operator uses ####################################

####### The .loc[] operator uses ####################################
# selecting a row by a label (the first coloumn starting from 0,... is the pandas indexing coloumn)
label_row   =   df.loc[2]
# print(label_row)

# selecting multiple rows by passing a list of labels
mult_rows   =   df.loc[[0,3,25]]
# print(mult_rows)

# selecting specific rows and coloumns
sub_data    =   df.loc[[0,3,25],["Glucose","Age"]]
# print(sub_data)
####### The .loc[] operator uses ####################################

##### filtering data to choose a subset which meets some condition ###########################
### choosing specific rows for which parameters meet some condition ####################

mask1   =   (df["Pregnancies"]<3)                       # this is a mask that, when passed in df[mask], will choose
                                                        # those rows for which pregnancies < 3
mask2   =   (df["Pregnancies"]<3) & (df["Age"]<30)      # this mask does something same but for 2 conditions

print(df[mask1])
print(df[mask2])

##### filtering data to choose a subset which meets some condition ###########################
### choosing specific rows for which parameters meet some condition ####################