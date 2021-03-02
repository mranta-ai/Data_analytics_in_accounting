## Example for structured data

import xgboost
import os
import pandas as pd
import matplotlib.pyplot as plt

#path = "C:/Users/mran/OneDrive - University of Vaasa/Data\TobinQ_Board/"
path = "C:/Users/mikko/OneDrive - University of Vaasa/Data\TobinQ_Board/"
os.chdir(path)

errors_df = pd.read_csv("AbnormalQ.csv",delimiter=";")

errors_df

y_df = errors_df['ABN_TOBIN_PANEL']
y_df = y_df.fillna(y_df.mean())

x_df = errors_df[['BOARD_GENDER_DIVERSITY_P',
       'BOARD_MEETING_ATTENDANCE', 'BOARD_MEMBER_AFFILIATION',
       'BOARD_MEMBER_COMPENSATIO', 'BOARD_SIZE', 'BOARD_SPECIFIC_SKILLS_PE',
       'CEO_BOARD_MEMBER', 'CHAIRMAN_IS_EX_CEO', 'EXECUTIVE_MEMBERS_GENDER',
       'INDEPENDENT_BOARD_MEMBER', 'NON_EXECUTIVE_BOARD_MEMB',
       'NUMBER_OF_BOARD_MEETINGS', 'STRICTLY_INDEPENDENT_BOA',
       'AVERAGE_BOARD_TENURE']]
x_df = x_df.fillna(x_df.mean())

dtrain = xgboost.DMatrix(x_df, label=y_df)

#param = {'max_depth': 1, 'eta': 0.9}
#param = {}
param = {'max_depth': 5, 'eta': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}

## Cross-validation

temp = xgboost.cv(param,dtrain,num_boost_round=400,nfold=5,seed=10)

plt.plot(temp['test-rmse-mean'][50:400])

## 120 boosting rounds selected. Then the fine tuning of tree parameters.
b_rounds = 150
m_depth = 5
eta = 0.1
ssample = 0.8
col_tree = 0.8
m_child_w = 1
gam = 0.0

#from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search

# Fine tune gamma

#param_test3 = {'gamma':[i/10.0 for i in range(0,40)]}
#super_param = {'gamma':[i/5.0 for i in range(0,11)],'min_child_weight':range(1,8,1),
#               'subsample':[i/20.0 for i in range(15,20)],'colsample_bytree':[i/20.0 for i in range(15,20)]}

#super_param

#gsearch3 = GridSearchCV(estimator = xgboost.XGBRegressor(njobs = -1, learning_rate = eta, n_estimators=b_rounds, max_depth=m_depth,
#     min_child_weight=m_child_w, gamma = gam, subsample=ssample, colsample_bytree=col_tree,objective='reg:squarederror',seed = 12),
#            n_jobs = -1, param_grid = param_test3,iid=False, cv=5)
#gsearch3 = GridSearchCV(estimator = xgboost.XGBRegressor(njobs = -1, learning_rate = eta, n_estimators=b_rounds, max_depth=m_depth,
#     min_child_weight=m_child_w, gamma = gam, subsample=ssample, colsample_bytree=col_tree,objective='reg:squarederror',seed = 12),
#            n_jobs = -1, param_grid = super_param,iid=False, cv=5)

#gsearch3.fit(x_df,y_df)

#gsearch3.best_params_

#gsearch3.best_score_

#gam = 2.0

# Fine tune max_depth and min_child_weight

#param_test1 = {'min_child_weight':range(1,9,1),'n_estimators':range(0,200,10)}
#param_test1 = {'max_depth':range(1,5,1),'min_child_weight':range(1,9,1)}
#param_test1 = {'max_depth':range(1,5,1)}
param_test1 = {'n_estimators':range(0,200,10)}

gsearch1 = GridSearchCV(estimator = xgboost.XGBRegressor(njobs = -1, learning_rate = eta, n_estimators=b_rounds, max_depth=m_depth,
     min_child_weight=m_child_w, gamma = gam,subsample=ssample, colsample_bytree=col_tree,objective='reg:squarederror', seed = 10),
        n_jobs = -1, param_grid = param_test1,iid=False, cv=5)

gsearch1.fit(x_df,y_df)

gsearch1.best_params_

gsearch1.best_score_

m_child_w = 1
m_depth = 4

# Subsample and col_sample_bytree

param_test4 = {'subsample':[i/20.0 for i in range(15,20)],'colsample_bytree':[i/20.0 for i in range(15,20)]}

param_test4

gsearch4 = GridSearchCV(estimator = xgboost.XGBRegressor(njobs = -1, learning_rate = eta, n_estimators=b_rounds, max_depth=m_depth,
     min_child_weight=m_child_w, gamma = gam, subsample=ssample, colsample_bytree=col_tree,objective='reg:squarederror', seed = 12),
            n_jobs = -1, param_grid = param_test4,iid=False, cv=10)

gsearch4.fit(x_df,y_df)

gsearch4.best_params_

gsearch4.best_score_

ssample = 0.81
col_tree = 0.86

# Again optimizing learning rate and boosting rounds

b_rounds = 200
m_depth = 6
eta = 0.05
ssample = 0.8
col_tree = 0.8
m_child_w = 6
gam = 0

param = {'max_depth': m_depth, 'eta': eta, 'subsample': ssample, 'colsample_bytree': col_tree, 'min_child_weight' : m_child_w, 'gamma' : gam}

temp = xgboost.cv(param,dtrain,num_boost_round=1000,nfold=5,seed = 12)

plt.plot(temp['test-rmse-mean'][200:600])

# BEST MODEL AT THE MOMENT
#b_rounds = 400
#m_depth = 6
#eta = 0.05
#ssample = 0.8
#col_tree = 0.8
#m_child_w = 6
#gam = 0

bst = xgboost.train(param,dtrain,num_boost_round=400)

split_bars = bst.get_split_value_histogram(feature='BOARD_GENDER_DIVERSITY_P')
plt.bar(split_bars['SplitValue'],split_bars['Count'])



