import matplotlib.pyplot as plt
import seaborn as sn
from model_prep import ModelPrep 
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
plt.style.use('seaborn')

#scicit packages
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,learning_curve
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error

# load model; set efficiency to true to get hourly averaged data set
data = ModelPrep(efficiency=None,sampler=None)

# predict max. outcome
data_calc = data.calc_power(c_p=0.59)
data_calc_df = pd.DataFrame(data_calc,columns=['power_pred'])
data_calc_df = data_calc_df.set_index(data.data.index)

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
window = 1
x_wind_speed_rolled = x_wind_speed.rolling(window=window, center=False).mean()
x_wind_dirx_rolled = x_wind_dir_x.rolling(window=window, center=False).mean()
x_wind_diry_rolled = x_wind_dir_y.rolling(window=window, center=False).mean()

#keras input data, multivariate input
all_data = pd.concat([x_wind_dir_x[window-1:-1],
                     x_wind_dir_y[window-1:-1],
                     x_pressure[window-1:-1],
                     data_calc_df[window-1:-1],
                     x_day[window-1:-1], 
                     x_temperature[window-1:-1],
                     y_power_measured[window-1:-1]],
                    axis=1)

# get training and test data

# arrange data for input layers and split in train, validation and test sets
test_set_size = 0.2 #10% test
valid_set_size= 0.2 #20% validation

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

##input data print and check dims
print('X_train:',X_train.shape)
print('X_val:' ,X_valid.shape)
print('X_test:',X_test.shape)
#
## y-data print and check dims
print('y_train:',y_train.shape)
print('y_val' ,y_valid.shape)
print('y_test:',y_test.shape)

## normalize X-data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)

# normalize y-data
standardize = MinMaxScaler()
standardize.fit(y_train)
y_train = standardize.transform(y_train)
y_valid = standardize.transform(y_valid)
y_test  = standardize.transform(y_test)


#set up model deepNL with 1 layer à 64 neurons, relu/sigmoid activiation function
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    #keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(8,activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

# optimizer
opt = keras.optimizers.Adam(learning_rate=0.001)

# compile model with best suitable loss function MSE and learning rate decay
model.compile(optimizer=opt,
              loss='mean_squared_logarithmic_error',
              metrics=['accuracy']
                            )
#parameter to change
epochs = 30
batch_size = 2

history = model.fit(X_train, y_train,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(X_valid, y_valid))
                    
path = 'figures/'

# plot loss
fig1 = plt.figure(figsize=(16, 9))
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs_plot = range(1,epochs+1)
plt.plot(epochs_plot, loss_train, 'g', label='Training loss')
plt.plot(epochs_plot, loss_val, 'b',   label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
fig1.savefig(path+'Loss',dpi=150,bbox_inches='tight') 

#plot accuracy
fig2 = plt.figure(figsize=(16, 9))
loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs_plot = range(1,epochs+1)
plt.plot(epochs_plot, loss_train, 'g', label='Training accuracy')
plt.plot(epochs_plot, loss_val, 'b',   label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
fig2.savefig(path+'Accuracy',dpi=150,bbox_inches='tight') 


# predict for test data
y_pred = model.predict(X_test)

# rescale if needed (output scaled?)
y_pred_rescaled = standardize.inverse_transform(y_pred)

#plot predictions vs. real data
fig3 = plt.figure(figsize=(16, 5))
plt.plot(df_test['power'].values,label='Original')
plt.plot(y_pred_rescaled,label='Predictions',linewidth=0.9,alpha=0.9)
plt.legend()
plt.show()
fig3.savefig(path+'predictions',dpi=150,bbox_inches='tight') 



# Printing error section and history
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse  = mean_absolute_error(y_test,y_pred)
print('Test RMSE:',(rmse))
print('MLP MSE:',(mse))

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = 'output/history_withdropout.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)