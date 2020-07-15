import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

path='ml/country_data.csv'
ds = pd.read_csv(path)
ds.head(13)

ds_x = ds['id']
ds_y = ds['totalcase']

x, x_test_patient, y, y_test_patient = train_test_split(ds_x, ds_y, test_size=0.2)

# x_test_patient = ds['num_of_date_test'][:12]
# y_test_patient = ds['num_of_patients_test'][:12]
x_prediction =[[140],[141],[142],[143],[144],[145],[146]]

yy=np.log10(y)

"""## **MLPRegressor**

### Code & Validation
"""

scores_1 = []
scores_2 = []
scores_3 = []
scores_4 = []
scores_5 = []
######################################################################################################################
MLP_Regressor_1 = MLPRegressor(hidden_layer_sizes=(4), activation='tanh', solver='lbfgs' ,learning_rate_init=0.01, max_iter=1000,random_state=120, validation_fraction=0.1)
MLP_Regressor_2 = MLPRegressor(hidden_layer_sizes=(5), activation='tanh', solver='lbfgs' ,learning_rate_init=0.01, max_iter=500,random_state=120, validation_fraction=0.1)
MLP_Regressor_3 = MLPRegressor(hidden_layer_sizes=(1), activation='tanh', solver='lbfgs' ,learning_rate_init=0.3, max_iter=1000,random_state=120, validation_fraction=0.2)
MLP_Regressor_4 = MLPRegressor(hidden_layer_sizes=(5), activation='relu', solver='lbfgs' ,learning_rate_init=0.01, max_iter=1000,random_state=1, validation_fraction=0.1)
MLP_Regressor_5 = MLPRegressor(hidden_layer_sizes=(5), activation='tanh', solver='sgd' ,learning_rate_init=0.01, max_iter=1000,random_state=1, validation_fraction=0.1)
######################################################################################################################
cv = KFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in cv.split(x):
    X_train, X_test, y_train, y_test ,yy_train, yy_test= x[train_index], x[test_index], y[train_index], y[test_index], yy[train_index], yy[test_index]
    #
    MLP_Regressor_1.fit(X_train.values.reshape(-1,1), yy_train)
    scores_1.append(MLP_Regressor_1.score(X_test.values.reshape(-1,1), yy_test))
    #
    MLP_Regressor_2.fit(X_train.values.reshape(-1,1), yy_train)
    scores_2.append(MLP_Regressor_2.score(X_test.values.reshape(-1,1), yy_test))
    #
    MLP_Regressor_3.fit(X_train.values.reshape(-1,1), yy_train)
    scores_3.append(MLP_Regressor_3.score(X_test.values.reshape(-1,1), yy_test))
     #
    MLP_Regressor_4.fit(X_train.values.reshape(-1,1), yy_train)
    scores_4.append(MLP_Regressor_4.score(X_test.values.reshape(-1,1), yy_test))
    #
    MLP_Regressor_5.fit(X_train.values.reshape(-1,1), yy_train)
    scores_5.append(MLP_Regressor_5.score(X_test.values.reshape(-1,1), yy_test))

print("Average score for MLP_Regressor_1:",sum(scores_1)/10,"\nAverage score for MLP_Regressor_2:",sum(scores_2)/10,"\nAverage score for MLP_Regressor_3:",sum(scores_3)/10
      ,"\nAverage score for MLP_Regressor_4:",sum(scores_4)/10,"\nAverage score for MLP_Regressor_5:",sum(scores_5)/10)

"""**After the validation we chosed best parameter for MLP (MLP_Regressor_2) to evaluate:**

### Evaluation
"""

MLP_Regressor = MLPRegressor(hidden_layer_sizes=(1), activation='tanh', solver='lbfgs' ,learning_rate_init=0.01, max_iter=1000,random_state=120, validation_fraction=0.1)
MLP_Regressor.fit(x.values.reshape(-1,1), yy)
y_test_patient_log=np.log10(y_test_patient)
evaluation_3 =MLP_Regressor.predict(x_test_patient.values.reshape(-1,1))
score=MLP_Regressor.score(x_test_patient.values.reshape(-1,1), y_test_patient_log)   
print("Final Evaluation Score for MLP_Regressor :",score)

"""### Prediction for new days"""

print('Evaluation for expecting 6 days in future in MLP_Regressor:')
for predict in x_prediction:
  print('day',predict,'=',int(10**MLP_Regressor.predict([predict])))

"""**Active Cases**

day [95] = 783429

day [96] = 807540

day [97] = 808516

day [98] = 827574

day [99] = 847802

day [100] = 870874

day [101] = 895106

### plot

Here the data is displayed by exponential
"""

predicted3 =10**MLP_Regressor.predict(x.values.reshape(-1,1))
predicted33=MLP_Regressor.predict(x.values.reshape(-1,1))
plt.plot(x, y, 'o-',label='data set')
plt.plot(x, predicted3,c='#ff0000',label='MLP_regression')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value ')
plt.show()

plt.plot(x, predicted3,label='MLP_regression')
plt.scatter(x_test_patient, 10**evaluation_3 ,s=50, c='#ff0000',label='Predict values')
plt.scatter(x_test_patient, y_test_patient,s=40, c='#003300',label='Original test values')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in Log')
plt.show()
