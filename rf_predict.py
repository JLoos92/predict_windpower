from model_prep import ModelPrep
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error,make_scorer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from pandas.plotting import lag_plot,autocorrelation_plot

# data from class
data = ModelPrep(efficiency=None,sampler=None)

# predict max. outcome
data_calc = data.calc_power(c_p=0.59)
data_calc_df = pd.DataFrame(data_calc,columns=['power_pred'])
data_calc_df = data_calc_df.set_index(data.data.index)

# seasonality


# input data seperated
y_power_measured = data.power_measured
x_wind_dir_x     = data.wind_x
x_wind_dir_y     = data.wind_y
x_temperature    = data.temperature
x_wind_speed     = data.wind_speed
x_pressure       = data.pressure
x_month          = data.month
x_hour           = data.hour
x_day            = data.day
x_winddir        = data.new_wdr

# one hot
x_hour_hot = data.hour_hot
x_day_hot  = data.day_hot
x_month_hot = data.month_hot
x_quarter_hot = data.quarter_hot
x_minute_hot  = data.minute_hot

# rolling mean for windspeed
window = 2
x_wind_speed_rolled = x_wind_speed.rolling(window=window, center=False).mean()
x_wind_dirx_rolled = x_wind_dir_x.rolling(window=window, center=False).mean()
x_wind_diry_rolled = x_wind_dir_y.rolling(window=window, center=False).mean()

#keras input data, multivariate input
all_data = pd.concat([x_wind_dir_x[window-1:-1],
                     x_wind_dir_y[window-1:-1],
                     x_winddir[window-1:-1],
                     x_month_hot[window-1:-1],
                     x_minute_hot[window-1:-1],
                     x_hour_hot[window-1:-1],
                     x_day_hot[window-1:-1],
                     x_quarter_hot[window-1:-1],
                     x_temperature[window-1:-1],
                     x_pressure[window-1:-1], 
                     data_calc_df[window-1:-1],
                     y_power_measured[window-1:-1]],
                    axis=1)
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

# append for gridsearch
train_x_cv = np.concatenate((train_set[0],valid_set[0]),axis=0)
train_y_cv = np.concatenate((train_set[1],valid_set[1]),axis=0)
tscv = TimeSeriesSplit(n_splits=4)

# random forest model scores
def rfr_model(X, y, kick_val=False):

    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(6,7,8),
            'n_estimators':range(5,15),
            'max_features': ['auto'],
            'min_samples_leaf' : range(29,35),
            'min_samples_split': (5,10,15)
        },
        cv=tscv, 
        verbose=0,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        return_train_score=True)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    model = RandomForestRegressor(max_depth=best_params["max_depth"],
     n_estimators=best_params["n_estimators"],
     min_samples_leaf=best_params["min_samples_leaf"],
     min_samples_split=best_params["min_samples_split"],
     random_state=False, 
     verbose=False)

    # Perform K-Fold CV
    #if kick_val == True:
    #    scores = cross_val_score(model, X, y, cv=10)
    #else:
    #scores = cross_val_score(model, valid_set[0], valid_set[1], cv=10, scoring='neg_root_mean_squared_error')

    return model, gsc, best_params

# make gridsearch (add valdata to train data)
model_params = rfr_model(train_x_cv,train_y_cv)

# get model
model = model_params[0]
gsc_results = model_params[1]

# fit model
fitted_model = model.fit(train_x_cv,train_y_cv)

#validset
test_x = test_set[0]
test_y = test_set[1]

#######################################################################################
# Plot scoring

train_scores_mean = gsc_results.cv_results_["mean_train_score"]
train_scores_std = gsc_results.cv_results_["std_train_score"]
test_scores_mean = gsc_results.cv_results_["mean_test_score"]
test_scores_std = gsc_results.cv_results_["std_test_score"]

plt.figure()
plt.title('Model')
plt.xlabel('estimators')
plt.ylabel('Score')
# plot train scores
plt.semilogx(train_scores_mean, label='Mean Train score',
             color='navy')
# create a shaded area between [mean - std, mean + std]
plt.gca().fill_between(train_scores_mean - train_scores_std,
                       train_scores_mean + train_scores_std,
                       alpha=0.2,
                       color='navy')
plt.semilogx(test_scores_mean,
             label='Mean Test score', color='darkorange')

# create a shaded area between [mean - std, mean + std]
plt.gca().fill_between(test_scores_mean - test_scores_std,
                       test_scores_mean + test_scores_std,
                       alpha=0.2,
                       color='darkorange')

plt.legend(loc='best')
plt.show()

#######################################################################################


# make prediction
yhat = model.predict(test_x)

# mean absolute error
errors = abs(yhat - test_set[1])
print('Mean Absolute Error:', round(np.mean(errors), 3), 'kW.')

# mape
mape = 100 * (errors / test_set[1])
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# r2 score
r2_score = r2_score(test_y,yhat)
print('R2-score:',100 * round(r2_score,4),'%.')

# RMSE and MSE
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(test_y, yhat)))
print('Mean Squared Error:', mean_squared_error(test_y, yhat)) 
print('Best params:',model[3]) 

# seaborn
plt.style.use('seaborn')

#plot predictions vs. measured
fig3 = plt.figure(figsize=(16, 5))
plt.plot(test_y,label='Original',linewidth=0.8)
plt.plot(yhat,label='Predictions',linewidth=0.9,alpha=0.9)
plt.legend()
plt.show()
 