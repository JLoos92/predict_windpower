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
model = ModelPrep(efficiency=False)

#input features loaded from model
x_wind_speed     = model.wind_speed
x_wind_direction = model.wind_direction
x_temperature    = model.temperature #minor
x_pressure       = model.pressure #minor  
yenerg           = model.power_measured
x_air_rho        = model.air_rho #minor

# vector fo windspeed in x and y direction
wind_x           = model.wind_x
wind_y           = model.wind_y

#input matrix
x_data = pd.concat([x_wind_speed,x_wind_direction],axis=1).values
x_data = x_data.astype(float)

#keras input data, univariate input
all_data = pd.concat([x_wind_speed,wind_x,wind_y,x_air_rho,yenerg],axis=1)

# get training and test data

# arrange data for input layers and split in train, validation and test sets
test_set_size = 0.1 #10% test
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
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

# compile model with best suitable loss function MSE and learning rate decay
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy']
                            )
#parameter to change
epochs = 40
batch_size = 5

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
loss_train = history.history['acc']
loss_val = history.history['val_acc']
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