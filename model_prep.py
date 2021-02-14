import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os 
from sklearn.metrics import mean_squared_error

# tex-style figure layout
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
pd.options.display.float_format = '{:.2f}'.format

class ModelPrep:
    '''
    '''

    def __init__(self,
                 air_p = None,
                 efficiency = None,
                 sampler = None):
    
        '''
    
        '''
    
        path = "input/data.csv"
        
        
        # set of parameters which are not included and which might be set 
        # manually
        
        RS = 287.058   #gas constante for DRY air
        
        # sampler is by default False
        # possibel inputs = H,D string
        self.sampler = sampler        
        self.air_p   = air_p
        self.efficiency = efficiency
        
        # pandas read csv columnwise
        self.data_check = pd.read_csv(path,sep=',')
        
        # fix data 
        self.data = self.data_check.groupby(self.data_check['time']).mean()
        self.data = self.data.reset_index()
             
        # convert to daytime-format       
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data.index = self.data['time']
        

        # hourly mean of each day in a year
        if self.efficiency == True:
         self.data = self.data.resample(sampler).mean()
        
        # check if NaN in dataframe if true linearly interpolate       
        if self.data.isnull().values.any() == True:
            self.data = self.data.interpolate()
        
        
        # raw parameters from data.csv; we make use of the class object to 
        # to call seperate parameters of the original data set
        #self.date_time      = self.data['time']        
        self.wind_direction = self.data['wind_direction'] 
        self.wind_speed     = self.data['wind_speed']
        self.temperature    = self.data['temperature']
        self.pressure       = self.data['pressure'] #already in Pa
        self.power_measured = self.data['power'] #/1e3 # convert to MW
        self.month          = self.data['time'].dt.month
        self.hour           = self.data['time'].dt.hour
        self.day            = self.data['time'].dt.day
        self.quarter        = self.data['time'].dt.quarter
        self.minute         = self.data['time'].dt.minute    
        
        # one hot for time series
        self.month_hot = pd.get_dummies(self.month)
        self.hour_hot  = pd.get_dummies(self.hour)
        self.day_hot   = pd.get_dummies(self.day)
        self.quarter_hot = pd.get_dummies(self.quarter)
        self.minute_hot  = pd.get_dummies(self.minute)

        # extra variables wind vectors
        # Convert to radians.
        wd_rad = self.wind_direction*np.pi / 180
        self.wind_x = self.wind_speed*np.cos(wd_rad)
        self.wind_y = self.wind_speed*np.sin(wd_rad)
       
        # simplified wind directions
        cut_labels = ['N', 'NE', 'E', 'SE' ,'S','SW','W','NW']
        cut_bins = [0, 45 ,90, 135, 180, 225, 270, 315, 360]
        self.new_wdr = pd.cut(self.wind_direction, bins=cut_bins, labels=cut_labels)
        self.new_wdr  = pd.get_dummies(self.new_wdr,prefix='wdr')     

        # calculate air density with PV = mRT // air_rho = P/RT
        self.RT = self.temperature * RS        #        
        self.air_rho = self.pressure/self.RT
        
        
        
        # only for testing! not used - but might be suitable for another input
    def calc_power(self,
                   c_p = None):
        
        '''
        '''
        self.c_p = c_p # betz limit at 0.59% of energy
                
        #Power = 0.5 x Swept Area x Air Density x Velocity3
        # given air density (calculated) wind speed and swept area, wind energy
        # can be calculated as following: 
        # wind_energy_predicted = 0.5 x Swept Area x Air Density x Velocity^3
        # here we assume a swept area of 10000
        
        self.power_predicted = 0.5 * 15.0 * self.air_rho * self.wind_speed**3
        self.power_predicted = self.power_predicted * self.c_p
    
        # faster numpy arrays
        self.measured = self.power_measured.to_numpy()
        self.predicted = self.power_predicted.to_numpy()
        
        return self.predicted
        









