#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:19:51 2021

@author: jo
"""

import numpy as np
import pandas as pd
import seaborn as sns

# scikit-learn modules: 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, KBinsDiscretizer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# # explicitly require this experimental feature
# from sklearn.experimental import enable_halving_search_cv # noqa
# # now you can import normally from model_selection
# from sklearn.model_selection import HalvingGridSearchCV



data = (pd.read_csv('input/data.csv',
                    parse_dates=['time'])
        # .head(1000)
        )

data = (data
        .groupby('time', as_index=False, sort=True)
        .apply(lambda x: (x.loc[x['ptime']==x['ptime'].max(),:]))
        .set_index('time')
        .drop(columns='ptime')
        )


# Wind x and y fractions:
wd_rad = data['wind_direction']*np.pi/180
data['wind_x'] = data['wind_speed']*np.cos(wd_rad)
data['wind_y'] = data['wind_speed']*np.sin(wd_rad)

RS = 287.058   #gas constante for DRY air
data['RT'] = data['temperature']*RS
data['air_rho'] = data['pressure']/data['RT']
data['sum_wind_last_3_hours'] = (data['wind_speed']
                                 .rolling('5H', min_periods=20)
                                 .sum())

data = data.loc[~data.isna().any(axis=1),:]
nans = data[data.isna().any(axis=1)]

# pairplot = sns.pairplot(data, height=3)
# pairplot.savefig('pairplot.png')

X = data.drop(columns='power')
y = data['power']


def binning(df, **kwargs):
    return df.apply(pd.cut, axis=0, **kwargs)

winddir_discretizer = Pipeline(
            [('binning', FunctionTransformer(
                binning, kw_args={
                    'bins': [0,45,90,135,180,225,270,315,360],
                    'retbins': False})),
              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('minmax', MinMaxScaler(), ['sum_wind_last_3_hours']),
        ('winddir_discreizer', winddir_discretizer, ['wind_direction'])
        ],
    remainder=StandardScaler()
    )

pipe = Pipeline(
    [('preprocess', preprocessor),
     ('forest', RandomForestRegressor(max_depth=70,
                                      n_estimators=200,
                                      n_jobs=-1))]
    )


param_grid = {
    'forest__max_depth': [20,100],
    'forest__n_estimators':[200,1000]
    }

model = GridSearchCV(pipe,
                     param_grid,
                     n_jobs=-1)

scores = cross_val_score(model,
                         X=X,
                         y=y,
                         scoring='neg_root_mean_squared_error',
                         cv=TimeSeriesSplit)
