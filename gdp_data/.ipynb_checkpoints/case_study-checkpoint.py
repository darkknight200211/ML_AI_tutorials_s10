import sklearn
import numpy as np
import matplotlib.pyplot as plt
import ML_user_def_functions as ML_fns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

########### getting the data of the gdp of a country over the years ###########################################
path_to_data = "/home/jwalitnpanchal/college_work/sem10/ML_AI/Gross domestic product (GDP) and main components per capita (nama_10_pc).csv"
data_frame = pd.read_csv(path_to_data)
# print(data_frame)
# print(type(data_frame.loc[4]))
years   =   data_frame.columns[4:]
# print(years)
years   =   [float(ele) for ele in years]
years   =   np.asarray(years).reshape(-1,1)
# print(years)

gdp_g   =   data_frame.loc[4].values[4:].astype(float)
# print(gdp_g)

figsize =   (5,3)
plt.figure(1, figsize=figsize)
plt.scatter(years, gdp_g, s=2.5)
plt.xlabel("years", fontsize=13)
plt.ylabel("gdp", fontsize=13)
plt.tight_layout()
plt.show(block=False)
########### getting the data of the gdp of a country over the years ###########################################

##### doing the regular gradient descent on this data and using sklearn linear regression model ##############
# learn_rate      =   0.001
# epoch           =   1000
# tol             =   1e-7
# _, _, y_pred    =   ML_fns.r_gradient_descent_linear_r(x_data=years, y_data=gdp_g, learning_rate=learn_rate, epoch=epoch, tol=tol, method="direct")
# print(y_pred)

model           =   LinearRegression()
model.fit(years, gdp_g)
y_pred_sklearn  =   model.predict(years)
r_sq            =   sklearn.metrics.r2_score(gdp_g, y_pred_sklearn)
plt.figure(2, figsize=figsize)
plt.plot(years, y_pred_sklearn, lw=2, color='red', label=f"{r_sq:.3f}")
plt.scatter(years, gdp_g, s=3.5)
plt.xlabel("years", fontsize=13)
plt.ylabel("gdp", fontsize=13)
plt.legend(fontsize=13, frameon=False)
plt.tight_layout()
plt.show(block=False)
##### doing the regular gradient descent on this data and using sklearn linear regression model ##############

############## multiple linear reg. ###################################
path_to_second_data     =       "/home/jwalitnpanchal/college_work/sem10/ML_AI/gdp_data/sorghum_annual_supply_disappearance.csv"
df2                     =       pd.read_csv(path_to_second_data)
print(df2,"\n",df2.info())
import_data             =       df2["imports"].values
import_data             =       np.asarray(import_data).reshape(-1,1)
# print(import_data)
export_data             =       np.asanyarray(df2["exports"].values).reshape(-1,1)
domestic_consume        =       np.asarray(df2["total_domestic_use"].values)
#### fitting model ####################
model2                  =       LinearRegression()
model2.fit(import_data, domestic_consume)
pred_dom_con_from_import    =   model2.predict(import_data)
r_sq_2                  =   sklearn.metrics.r2_score(domestic_consume, pred_dom_con_from_import)
f1, p1                  =   f_regression(X=export_data, y=domestic_consume)
print(f1)

model3                  =       LinearRegression()
model3.fit(export_data, domestic_consume)
pred_dom_con_from_export    =   model3.predict(export_data)
r_sq_3                  =   sklearn.metrics.r2_score(domestic_consume, pred_dom_con_from_export)

#### fitting model ####################
# print(len(import_data), len(export_data), len(domestic_consume))
fig, axs                =       plt.subplots(1,2,figsize=figsize, sharey=True)
axs[0].plot(import_data, pred_dom_con_from_import,lw=2.0, label=f"r-sq={r_sq_2:.3f}")
axs[0].scatter(import_data, domestic_consume, color="red", s=2.5)
axs[0].set_xlabel("import", fontsize=13)
axs[0].set_ylabel("consumption", fontsize=13)
axs[1].plot(export_data, pred_dom_con_from_export,lw=2.0,label=f"r-sq={r_sq_3:.3f}")
axs[1].scatter(export_data, domestic_consume,color="red", s=2.5)
axs[1].set_xlabel("export", fontsize=13)
[axs[i].tick_params(axis='both', direction='in') for i in range(2)]
plt.tight_layout()
[axs[i].legend(fontsize=13, frameon=False) for i in range(2)]
plt.show(block=False)

## 3d figure ####
fig3d   =   plt.figure(4, figsize=figsize)
ax3d    =   fig3d.add_subplot(projection="3d")
ax3d.scatter(import_data, export_data, domestic_consume)
ax3d.set_xlabel('import')
ax3d.set_ylabel('export')
ax3d.set_zlabel('consuption')
plt.tight_layout()
plt.show()
############## multiple linear reg. ###################################