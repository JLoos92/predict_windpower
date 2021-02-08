from model_prep import ModelPrep
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# data from class
data = ModelPrep(efficiency=None,sampler=None)

# input data seperated
y_power_measured = data.power_measured
x_wind_dir_x     = data.wind_x
x_wind_dir_y     = data.wind_y
x_temperature    = data.temperature
x_wind_speed     = data.wind_speed
x_pressure       = data.pressure

#keras input data, univariate input
all_data = pd.concat([x_wind_speed,x_wind_dir_x, x_wind_dir_y,x_temperature,x_pressure ,y_power_measured], axis=1)
# define test and train data set

def split_train_test(test_set_size=0.2,valid_set_size=0.1):
    #split 
    df_test = all_data.iloc[ int(np.floor(len(all_data)*(1-test_set_size))) : ]
    df_train_plus_valid = all_data.iloc[ : int(np.floor(len(all_data)*(1-test_set_size))) ]

    df_train = df_train_plus_valid.iloc[ : int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) ]
    df_valid = df_train_plus_valid.iloc[ int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) : ]

    X_train, y_train = df_train.iloc[0:, :-1], df_train['power']
    X_valid, y_valid = df_valid.iloc[0:, :-1], df_valid['power']
    X_test, y_test = df_test.iloc[0:, :-1], df_test['power']

    # reshape and transpose for input layer; to numpy array
    X_train, y_train = X_train.values,y_train.values.reshape((1,len(y_train))).T
    X_valid, y_valid = X_valid.values,y_valid.values.reshape((1,len(y_valid))).T
    X_test, y_test   = X_test.values,y_test.values.reshape((1,len(y_test))).T

    return (X_train,y_train),(X_valid,y_valid),(X_test,y_test)


# get train,valid and test set
train_set = split_train_test()[0]
valid_set = split_train_test()[1]
test_set  = split_train_test()[2]

# random forest model scores
def rfr_model(X, y):

    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 500),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    model = RandomForestRegressor(max_depth=best_params["max_depth"],
     n_estimators=best_params["n_estimators"],
     random_state=False, verbose=False)

    # Perform K-Fold CV
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')
    return model, scores

model_params = rfr_model(train_set[0],train_set[1])

# get model
model = model_params[0]
fitted_model = model.fit(train_set[0],train_set[1])

#validset
valid_x = valid_set[0]
valid_y = valid_set[1]
# make prediction
yhat = model.predict(valid_x)

# mean absolute error
errors = abs(yhat - valid_set[1])
print('Mean Absolute Error:', round(np.mean(errors), 3), 'kW.')

# mape
mape = 100 * (errors / valid_set[1])
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# r2 score
r2_score = r2_score(valid_y,yhat)
print('R2-score:',100 * round(r2_score,4),'%.')
# show predicts vs. measured
plt.style.use('seaborn')

#plot predictions vs. real data
fig3 = plt.figure(figsize=(16, 5))
plt.plot(valid_y,label='Original')
plt.plot(yhat,label='Predictions',linewidth=0.9,alpha=0.9)
plt.legend()
plt.show()
 