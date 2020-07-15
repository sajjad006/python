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

path='https://raw.githubusercontent.com/AbedMHroub/Corona-virus-COVID19-predictions-project/master/dataset/Infected.csv'
ds = pd.read_csv(path, index_col=0)
ds.head(13)

x = ds['num_of_date']
y = ds['num_of_patients']
x_test_patient = ds['num_of_date_test'][:12]
y_test_patient = ds['num_of_patients_test'][:12]
x_prediction =[[95],[96],[97],[98],[99],[100],[101]]

"""## **linear_regression**

### Code & Validation
"""

linear_regression=LinearRegression()

yy=np.log10(y)

scores = []
######################################################################################################################
Linear_Regression = LinearRegression()
######################################################################################################################
cv = KFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in cv.split(x):
    X_train, X_test, y_train, y_test , yy_train, yy_test= x[train_index], x[test_index], y[train_index], y[test_index], yy[train_index], yy[test_index]
    Linear_Regression.fit(X_train.values.reshape(-1,1), yy_train)
    scores.append(Linear_Regression.score(X_test.values.reshape(-1,1), yy_test))
    
print("Average score for Linear Regression:",sum(scores)/len(scores))

"""**After the validation we evaluate the Linear Model:**

### Evaluation
"""

Linear_Regression.fit(x.values.reshape(-1,1), yy)
y_test_patient_log=np.log10(y_test_patient)
evaluation_1 =Linear_Regression.predict(x_test_patient.values.reshape(-1,1))
score=Linear_Regression.score(x_test_patient.values.reshape(-1,1), y_test_patient_log)   
print("Final Evaluation Score for Linear Regression :",score)

"""### Prediction for new days"""

print('Evaluation for expecting 6 days in future in Linear_Regression:')
for predict in x_prediction:
  print('day',predict,'=',int(10**Linear_Regression.predict([predict])))

"""### plot

Here the data are displayed in logarithmic representation
"""

predicted1 = Linear_Regression.predict(x.values.reshape(-1,1))
predicted11 = 10**Linear_Regression.predict(x.values.reshape(-1,1))
plt.plot(x, yy, 'o-',label='data set')
plt.plot(x, predicted1,c='#ff0000',label='linear_regression')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in log ')
plt.show()

"""---


Here the data is displayed by exponential representation (linear_regression Curve Was Converted)
"""

plt.plot(x[:79], predicted11[:79],label='linear_regression')
plt.scatter(x_test_patient, 10**evaluation_1 ,s=60, c='#ff0000',label='Predict values')
plt.scatter(x_test_patient, y_test_patient ,s=50, c='#003300',label='Original test values')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value ')
plt.show()

"""## **SVR**

### Code & Validation
"""

scores_1 = []
scores_2 = []
scores_3 = []
scores_4 = []
scores_5 = []
######################################################################################################################
SVR_Regressor_1 = SVR(kernel='rbf')
SVR_Regressor_2 = SVR(kernel='poly',degree=5)
SVR_Regressor_3 = SVR(kernel='linear')
SVR_Regressor_4 = SVR(kernel='poly',degree=4)
SVR_Regressor_5 = SVR(kernel='poly',degree=10)
######################################################################################################################
cv = KFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in cv.split(x):
    X_train, X_test, y_train, y_test ,yy_train, yy_test= x[train_index], x[test_index], y[train_index], y[test_index], yy[train_index], yy[test_index]
    #
    SVR_Regressor_1.fit(X_train.values.reshape(-1,1), y_train)
    scores_1.append(SVR_Regressor_1.score(X_test.values.reshape(-1,1), y_test))
    #
    SVR_Regressor_2.fit(X_train.values.reshape(-1,1), y_train)
    scores_2.append(SVR_Regressor_2.score(X_test.values.reshape(-1,1), y_test))
    #
    SVR_Regressor_3.fit(X_train.values.reshape(-1,1), y_train)
    scores_3.append(SVR_Regressor_3.score(X_test.values.reshape(-1,1), y_test))
    #
    SVR_Regressor_4.fit(X_train.values.reshape(-1,1), y_train)
    scores_4.append(SVR_Regressor_4.score(X_test.values.reshape(-1,1), y_test))
    #
    SVR_Regressor_5.fit(X_train.values.reshape(-1,1), y_train)
    scores_5.append(SVR_Regressor_5.score(X_test.values.reshape(-1,1), y_test))

print("Average score for SVR_Regressor_1:",sum(scores_1)/10,
      "\nAverage score for SVR_Regressor_2:",sum(scores_2)/10,
      "\nAverage score for SVR_Regressor_3:",sum(scores_3)/10,
      "\nAverage score for SVR_Regressor_4:",sum(scores_4)/10,
      "\nAverage score for SVR_Regressor_5:",sum(scores_5)/10)

"""**After the validation we chosed best parameter for SVR (SVR_Regressor_2) to evaluate:**

### Evaluation
"""

SVR_Regressor_2.fit(x.values.reshape(-1,1), y)
y_test_patient_log=np.log10(y_test_patient)
evaluation_2 =SVR_Regressor_2.predict(x_test_patient.values.reshape(-1,1))
score=SVR_Regressor_2.score(x_test_patient.values.reshape(-1,1), y_test_patient)   
print("Final Evaluation Score for SVR Regression :",score)

"""### Prediction for new days"""

print('Evaluation for expecting 6 days in future in SVR_Regression:')
for predict in x_prediction:
  print('day',predict,'=',int(SVR_Regressor_2.predict([predict])))

"""### plot

Here the data is displayed by exponential representation
"""

predicted2 = (SVR_Regressor_2.predict(x.values.reshape(-1,1)))
predicted22= np.log10(predicted2[23:])
plt.plot(x,y, 'o-',label='data set')
plt.plot(x, predicted2,c='#ff0000',label='SVR_Regressor_2')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in log')
plt.show()

plt.plot(x, predicted2,label='SVR_Regressor_2')
plt.scatter(x_test_patient, evaluation_2 ,s=60, c='#ff0000',label='Predict values')
plt.scatter(x_test_patient, y_test_patient ,s=50, c='#003300',label='Original test values')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in Log')
plt.show()

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

"""## **Conclusion**"""

print("\n\t   The Three models in exponantial case ")
plt.plot(x[:76], 10**predicted1[:76],label='linear_regression')
plt.plot(x, predicted2,label='SVR')
plt.plot(x, predicted3,label='MLPRegressor (THE BEST)')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value')
plt.show()

print("\n\t   The Three models in Log case ")
plt.plot(x, predicted1,label='linear_regression')
plt.plot(x[23:], predicted22,label='SVR')
plt.plot(x, predicted33,label='MLPRegressor (THE BEST)')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value')
plt.show()

print("\n\t   The Best model (MLP Model) ")
plt.plot(x, predicted3,label='MLPRegressor (THE BEST)')
plt.plot(x_prediction, 10**MLP_Regressor.predict(x_prediction),label='new day prediction')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value')
plt.show()

print("\n\t The Best model (MLP Model) in Log")
plt.plot(x, np.log10(predicted3),label='MLPRegressor (THE BEST)')
plt.plot(x_prediction, MLP_Regressor.predict(x_prediction), label='new day prediction')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in log')
plt.show()

"""# **Deaths**

## **Preparing deaths data**
"""

path2='https://raw.githubusercontent.com/AbedMHroub/Corona-virus-COVID19-predictions-project/master/dataset/Deaths_us.csv'
ds_death = pd.read_csv(path2, index_col=0)
ds_death.head(10)

x_death = ds_death['num_of_date']
y_death = ds_death['Deaths']
x_test_deaths = ds_death['num_of_date_test'][:8]
y_test_deaths = ds_death['Deaths_test'][:8]
x_prediction2 =[[60],[61],[62],[63],[64],[65],[67]]

"""## **linear_regression**

### Code & Validation
"""

from sklearn.linear_model import LinearRegression
linear_regression=LinearRegression()

y_log_D = np.log10(y_death)

scores = []
######################################################################################################################
Linear_Regression = LinearRegression()
######################################################################################################################
cv = KFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in cv.split(x_death):
    X_train_D, X_test_D, y_log_train_D, y_log_test_D= x_death[train_index], x_death[test_index], y_log_D[train_index], y_log_D[test_index]
    
    Linear_Regression.fit(X_train_D.values.reshape(-1,1), y_log_train_D)
    scores.append(Linear_Regression.score(X_test_D.values.reshape(-1,1), y_log_test_D))
    
print("Average score for Linear Regression:",sum(scores)/10)

"""**After the validation we evaluate the Linear Model:**

### Evaluation
"""

Linear_Regression.fit(x_death.values.reshape(-1,1), y_log_D)
y_test_deaths_log=np.log10(y_test_deaths)
evaluation_4 =Linear_Regression.predict(x_test_deaths.values.reshape(-1,1))
score=Linear_Regression.score(x_test_deaths.values.reshape(-1,1), y_test_deaths_log)   
print("Final Evaluation Score for Linear Regression :",score)

"""### Prediction for new days"""

print('Evaluation for expecting 6 days in future in Linear_Regression:')
for predict in x_prediction2:
  print('day',predict,'=',int(10**Linear_Regression.predict([predict])))

"""### plot

Here the data are displayed in logarithmic representation
"""

predicted4 =(Linear_Regression.predict(x_death.values.reshape(-1,1)))
plt.plot(x_death, y_log_D, 'o-',label='data set')
plt.plot(x_death, predicted4,c='#ff0000',label='linear_regression')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in Log')
plt.show()

"""---


Here the data are displayed in logarithmic representation
"""

plt.plot(x_death[:58], 10**(predicted4[:58]),label='linear_regression')
plt.scatter(x_test_deaths, 10**evaluation_4 ,s=60, c='#ff0000',label='Predict values')
plt.scatter(x_test_deaths, y_test_deaths ,s=50, c='#003300',label='Original values')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in Log')
plt.show()

"""## **SVR**

### Code & Validation
"""

from sklearn.svm import SVR


scores_1 = []
scores_2 = []
scores_3 = []
######################################################################################################################
SVR_Regressor_1 = SVR(kernel='rbf')
SVR_Regressor_2 = SVR(kernel='poly',degree=4)
SVR_Regressor_3 = SVR(kernel='linear')
######################################################################################################################
cv = KFold(n_splits=8, random_state=1, shuffle=True)
for train_index, test_index in cv.split(x_death):
    X_train_D, X_test_D, y_train_D, y_test_D = x_death[train_index], x_death[test_index], y_death[train_index], y_death[test_index]
    #
    SVR_Regressor_1.fit(X_train_D.values.reshape(-1,1), y_train_D)
    scores_1.append(SVR_Regressor_1.score(X_test_D.values.reshape(-1,1), y_test_D))
    #
    SVR_Regressor_2.fit(X_train_D.values.reshape(-1,1), y_train_D)
    scores_2.append(SVR_Regressor_2.score(X_test_D.values.reshape(-1,1), y_test_D))
    #
    SVR_Regressor_3.fit(X_train_D.values.reshape(-1,1), y_train_D)
    scores_3.append(SVR_Regressor_3.score(X_test_D.values.reshape(-1,1), y_test_D))

print("Average score for SVR_Regressor_1:",sum(scores_1)/8,"\nAverage score for SVR_Regressor_2:",sum(scores_2)/8,"\nAverage score for SVR_Regressor_3:",sum(scores_3)/8)

"""**After the validation we chosed best parameter for SVR (SVR_Regressor_2) to evaluate:**

### Evaluation
"""

SVR_Regressor_2.fit(x_death.values.reshape(-1,1), y_death)
y_test_deaths_log=np.log10(y_test_deaths)
evaluation_5 =SVR_Regressor_2.predict(x_test_deaths.values.reshape(-1,1))
score=SVR_Regressor_2.score(x_test_deaths.values.reshape(-1,1), y_test_deaths)   
print("Final Evaluation Score for SVR Regression :",score)

"""### Prediction for new days"""

print('Evaluation for expecting 6 days in future in SVR_Regression:')
for predict in x_prediction2:
  print('day',predict,'=',int(SVR_Regressor_2.predict([predict])))

"""### plot

Here the data are displayed in logarithmic representation
"""

predicted5 = (SVR_Regressor_2.predict(x_death.values.reshape(-1,1)))
plt.plot(x_death, y_death, 'o-',label='data set')
plt.plot(x_death, predicted5,c='#ff0000',label='SVR_Regressor_2')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in Log')
plt.show()

plt.plot(x_death, predicted5,label='SVR_Regressor_2')
plt.scatter(x_test_deaths, evaluation_5 ,s=60, c='#ff0000',label='Predict values')
plt.scatter(x_test_deaths, y_test_deaths ,s=50, c='#003300',label='Original test values')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in Log')
plt.show()

"""## **MLPRegressor**

### Code & Validation
"""

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

scores_1 = []
scores_2 = []
scores_3 = []
scores_4 = []
scores_5 = []

######################################################################################################################
MLP_Regressor_1 = MLPRegressor(hidden_layer_sizes=(4), activation='tanh', solver='lbfgs' ,learning_rate_init=0.01, max_iter=1000,random_state=93, validation_fraction=0.1)
MLP_Regressor_2 = MLPRegressor(hidden_layer_sizes=(5), activation='tanh', solver='lbfgs' ,learning_rate_init=0.01, max_iter=1000,random_state=93, validation_fraction=0.1)
MLP_Regressor_3 = MLPRegressor(hidden_layer_sizes=(6), activation='tanh', solver='lbfgs' ,learning_rate_init=0.01, max_iter=1000,random_state=93, validation_fraction=0.1)
MLP_Regressor_4 = MLPRegressor(hidden_layer_sizes=(4), activation='relu', solver='lbfgs' ,learning_rate_init=0.01, max_iter=1000,random_state=93, validation_fraction=0.1)
MLP_Regressor_5 = MLPRegressor(hidden_layer_sizes=(4), activation='tanh', solver='sgd' ,learning_rate_init=0.01, max_iter=1000,random_state=93, validation_fraction=0.1)
######################################################################################################################
cv = KFold(n_splits=8, random_state=1, shuffle=True)
for train_index, test_index in cv.split(x_death):
    X_train_D, X_test_D, y_log_train_D, y_log_test_D = x_death[train_index], x_death[test_index], y_log_D[train_index], y_log_D[test_index]
    #
    MLP_Regressor_1.fit(X_train_D.values.reshape(-1,1), y_log_train_D)
    scores_1.append(MLP_Regressor_1.score(X_test_D.values.reshape(-1,1), y_log_test_D))
    #
    MLP_Regressor_2.fit(X_train_D.values.reshape(-1,1), y_log_train_D)
    scores_2.append(MLP_Regressor_2.score(X_test_D.values.reshape(-1,1), y_log_test_D))
    #
    MLP_Regressor_3.fit(X_train_D.values.reshape(-1,1), y_log_train_D)
    scores_3.append(MLP_Regressor_3.score(X_test_D.values.reshape(-1,1), y_log_test_D))
    #
    MLP_Regressor_4.fit(X_train_D.values.reshape(-1,1), y_log_train_D)
    scores_4.append(MLP_Regressor_4.score(X_test_D.values.reshape(-1,1), y_log_test_D))
    #
    MLP_Regressor_5.fit(X_train_D.values.reshape(-1,1), y_log_train_D)
    scores_5.append(MLP_Regressor_5.score(X_test_D.values.reshape(-1,1), y_log_test_D))

print(  "Average score for MLP_Regressor_1:",sum(scores_1)/8,
      "\nAverage score for MLP_Regressor_2:",sum(scores_2)/8,
      "\nAverage score for MLP_Regressor_3:",sum(scores_3)/8,
      "\nAverage score for MLP_Regressor_4:",sum(scores_4)/8,
      "\nAverage score for MLP_Regressor_5:",sum(scores_5)/8)

"""**After the validation we chosed best parameter for MLP (MLP_Regressor_2) to evaluate:**

### Evaluation
"""

MLP_Regressor_D = MLPRegressor(hidden_layer_sizes=(5), activation='tanh', solver='lbfgs' ,learning_rate_init=0.01,
                               max_iter=1000,random_state=93, validation_fraction=0.1)
MLP_Regressor_D.fit(x_death.values.reshape(-1,1), y_log_D)
y_test_deaths_log=np.log10(y_test_deaths)
evaluation_6 =MLP_Regressor_D.predict(x_test_deaths.values.reshape(-1,1))
score=MLP_Regressor_D.score(x_test_deaths.values.reshape(-1,1), y_test_deaths_log)   
print("Final Evaluation Score for MLP Regression :",score)

"""### Prediction for new days"""

print('Evaluation for expecting 6 days in future in MLP_Regression:')
for predict in x_prediction2:
  print('day',predict,'=',int(10**MLP_Regressor_D.predict([predict])))

"""**Deaths**

day [60] = 63856

day [61] = 65753

day [62] = 67444

day [63] = 68597

day [64] = 69921

day [65] = 72271

day [67] = 74799

### plot

Here the data are displayed in logarithmic representation
"""

predicted6 = MLP_Regressor_D.predict(x_death.values.reshape(-1,1))
plt.plot(x_death, y_death, 'o-',label='data set')
plt.plot(x_death, 10**predicted6,c='#ff0000',label='MLP_Regressor_D')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in log')
plt.show()

plt.plot(x_death, 10**predicted6,label='MLP_Regressor_D')
plt.scatter(x_test_deaths, 10**evaluation_6 ,s=60, c='#ff0000',label='Predict values')
plt.scatter(x_test_deaths, y_test_deaths ,s=50, c='#003300',label='Original test values')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in Log')
plt.show()

"""## **Conclusion**"""

print("\n\t   The Three models in exponantial case ")
plt.plot(x_death[:45], 10**predicted4[:45],label='linear_regression')
plt.plot(x_death, predicted5,label='SVR')
plt.plot(x_death, 10**predicted6,label='MLPRegressor (THE BEST)')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value')
plt.show()

print("\n\t   The Three models in Log case ")
plt.plot(x_death, predicted4,label='linear_regression')
plt.plot(x_death, np.log10(predicted5),label='SVR')
plt.plot(x_death, predicted6,label='MLPRegressor (THE BEST)')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value')
plt.show()

print("\n\t   The Best model (MLP Model) in exponantial case ")
plt.plot(x_death, 10**predicted6,label='MLPRegressor (THE BEST)')
plt.plot(x_prediction2, 10**MLP_Regressor_D.predict(x_prediction2),label='new day prediction ')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value')
plt.show()

print("\n\t The Best model (MLP Model) in Log")
plt.plot(x_death, predicted6,label='MLPRegressor (THE BEST)')
plt.plot(x_prediction2, MLP_Regressor_D.predict(x_prediction2), label='new day prediction')
plt.legend()
plt.xlabel('Number of Day')
plt.ylabel('Predict value in log')
plt.show()