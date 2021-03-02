### Xgboost, state-of-the-art ensemble method for structured data

Xgboost is one of the latest iterations of gradient boosting algorithms, with many optimisations compared to other gradient boosting algorithms.  
https://xgboost.readthedocs.io/en/latest/

#### Libraries
*Pandas* and *Numpy* for data processing.  
*Scipy* and *Statsmodels* for statistical analysis.  
*Matplotlib* for plotting.  

import xgboost
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import scipy.stats as ss
import statsmodels.api as sm

A function to check the normality of a dataset. Draws a histogram and a normal curve. It calculates also the Kolmogorov-Smirnov statistic of the dataset.

# Function to check normality
def check_normality(data,bins=50):
    mask = ~np.isnan(data)
    temp1 = data[mask]
    x = np.linspace(np.percentile(temp1,1),np.percentile(temp1,99),100)
    plt.hist(temp1, density=True,bins=bins)
    plt.plot(x,ss.norm.pdf(x,np.mean(temp1),np.std(temp1)))
    print("Kolmogorov-Smirnov: " + str(sm.stats.diagnostic.kstest_normal(temp1)))

# Different folders, depending which computer I am using
path = "C:/Users/mran/OneDrive - University of Vaasa/Data\TobinQ_Board/"
#path = "C:/Users/mikko/OneDrive - University of Vaasa/Data\TobinQ_Board/"
#path = "E:/Onedrive_uwasa/OneDrive - University of Vaasa/Data\TobinQ_Board/"
os.chdir(path)

Data preparation, which includes:  
* Loading data from a csv-file
* Remove missing y-values
* Loading the abnormal Tobin Q values to the y_df dataframe
* Loading the board characteristic variables to the x_df dataframe
* Winsorizing the data, meaning that we change the most extreme values to the 1 % and 99 % percentile values.

# Prepare data.
errors_df = pd.read_csv("AbnormalQ.csv",delimiter=";")
errors_df = errors_df.rename(columns = {'BOARD_GENDER_DIVERSITY_P' : 'BOARD_GENDER_DIVERSITY','BOARD_MEMBER_COMPENSATIO' : 'BOARD_MEMBER_COMPENSATION'})
# Remove missing abQ values
errors_df = errors_df.loc[~errors_df['ABN_TOBIN_POIKKILEIK'].isna()]
y_df = errors_df['ABN_TOBIN_POIKKILEIK']
x_df = errors_df[['BOARD_GENDER_DIVERSITY',
       'BOARD_MEETING_ATTENDANCE', 'BOARD_MEMBER_AFFILIATION',
       'BOARD_MEMBER_COMPENSATION', 'BOARD_SIZE',
       'CEO_BOARD_MEMBER', 'CHAIRMAN_IS_EX_CEO',
       'INDEPENDENT_BOARD_MEMBER',
       'NUMBER_OF_BOARD_MEETINGS',
       'AVERAGE_BOARD_TENURE']]
# Winsorize x-data
x_df_wins = x_df.clip(lower=x_df.quantile(0.01), upper=x_df.quantile(0.99), axis = 1)

Calculate the descriptive statistics of the x variables (board characteristics).

x_df.describe().transpose()

Convert the data to a xgboost dmatrix, that is computationally efficient form for data.

dtrain = xgboost.DMatrix(x_df_wins, label=y_df, nthread = -1)

***
The most difficult part in using xgboost for prediction is the tuning of hyperparameters. There is no good theory to guide parameter optimisation, and it is more dark magic than science. You can use grid-search approaches, but it needs A LOT of computing power. The tunable parameters are:  
* num_boost_round: The number of decision trees.
* max_depth: The depth of the trees.
* eta: The weight of the added tree
* subsample: A randomly selected subsample of the data, that is used at each round.
* colsample_bytree: A randomly selected subsample of features, that is used for each tree (this can be done also by node and by level).
* min_child_weight: A regularisation parameter. In the linear regression case this would mean the minimum number of data points that needs to be in the leaves
* gamma: A regularisation parameter, that controls the loss function.

m_depth = 5
eta = 0.02
ssample = 0.8
col_tree = 0.8
m_child_w = 1
gam = 0.1
param = {'max_depth': m_depth, 'eta': eta, 'subsample': ssample, 'colsample_bytree': col_tree, 'min_child_weight' : m_child_w, 'gamma' : gam}

***
Xgboost.cv -function can be used to search the optimal number of trees in the model. We search the number of trees that achieves the lowest *root-mean-square-error*. In this case the optimal number of trees appeas to be around 600.

temp = xgboost.cv(param,dtrain,num_boost_round=900,nfold=5,seed=10)

plt.plot(temp['test-rmse-mean'][200:900])

b_rounds = 600

***
The model is trained using the *train*-function. The number of trees was set to 600 based on the cross-validation results.

bst = xgboost.train(param,dtrain,num_boost_round=b_rounds)

***
Histrogram for the split points. Notice that this information cannot be used for deciding what is the optimal percentage. The model just tries to separate companies according to their market performance. This is not necessarily achieved by separating data using the "optimal" feature values.

# Histogram of split points
split_bars = bst.get_split_value_histogram(feature='BOARD_GENDER_DIVERSITY',bins = None)
plt.bar(split_bars['SplitValue'],split_bars['Count'])
plt.xlabel('Gender diversity')
plt.ylabel('Count')
#plt.savefig('test.png',dpi=300)

***
Average of the split used in the model.

aver_split_points = []
for feat in bst.feature_names:
    split_bars = bst.get_split_value_histogram(feature=feat,as_pandas=False)
    temp = sum(split_bars[:,0]*split_bars[:,1])/(sum(split_bars[:,1]))
    aver_split_points.append(temp)
aver_df = pd.DataFrame()
aver_df['averages'] = aver_split_points
aver_df.index = bst.feature_names
aver_df

Basic metrics that can be used to measure the feature importance are *weight,gain* and *cover*. There are many issues with these metrics. For example the weight metric undervalues binary features. Below is their explanation from the xgboost documents:
* ‘weight’: the number of times a feature is used to split the data across all trees.
* ‘gain’: the average gain across all splits the feature is used in.
* ‘cover’: the average coverage across all splits the feature is used in.

xgboost.plot_importance(bst,importance_type='weight',xlabel='weight',show_values=False),
xgboost.plot_importance(bst,importance_type='gain',xlabel='gain',show_values=False),
xgboost.plot_importance(bst,importance_type='cover',xlabel='cover',show_values=False)

### SHAP analysis
Because the basic importance metrics have some many issues, the machine learning community is doing a lot of research at the moment to invent better ways to analyse feature importance. One recent innovation is SHAP values. https://github.com/slundberg/shap

j=0
shap.initjs()

Below, shap values are calculated for the model.

explainerXGB = shap.TreeExplainer(bst)
shap_values_XGB_test = explainerXGB.shap_values(x_df_wins,y_df,check_additivity = False)
#interaction_values = explainerXGB.shap_interaction_values(x_df_wins,y_df)

A summary plot of the most important features. From the board characteristics the most important explainers of market performance are *Number of board meetings*, *Board member compensation*, *Average board tenure* and *Board member affiliation*. In this example, the meaning of the SHAP values is *the absolute average effect on the abnormal tobin Q of the company*.

shap.summary_plot(shap_values_XGB_test,x_df_wins,plot_type='bar')

Present the average values as a dataframe.

shaps = np.mean(abs(shap_values_XGB_test), axis = 0)
names = bst.feature_names
apu_df = pd.DataFrame()
apu_df['names'] = names
apu_df['shaps'] = shaps
apu_df

The above SHAP plot has no direction information. *Number of board meetings* has the largest effect, but is it increasing or decreasing abnormal Tobin Q? This is estimated below.

stand_feats = (x_df_wins-x_df_wins.mean(axis = 0))/x_df_wins.std(axis = 0)
std_feat_times_shaps = np.multiply(shap_values_XGB_test,stand_feats)
dir_metric = np.mean(std_feat_times_shaps, axis = 0)
plt.barh(range(10),dir_metric,tick_label = bst.feature_names)

More detailed dependence can also be analysed using a scatter plot.

shap.dependence_plot(0,shap_values_XGB_test,x_df),
shap.dependence_plot(8,shap_values_XGB_test,x_df)

An example of the check_normality -function.

check_normality(y_df)

### GAM MODEL
Generalised additive models can be used to plot a nonlinear trendline above the scatter plots

from pygam import LinearGAM, s, f
from pygam.datasets import mcycle
X, y = mcycle(return_X_y=True)

fig, axs = plt.subplots(5,2,figsize=(10,14),squeeze=True)
ind = 0
for ax in axs.flat:
    feat = bst.feature_names[ind]
    temp_df = pd.DataFrame()
    temp_df['data'] = x_df_wins[feat]
    temp_df['shap'] = shap_values_XGB_test[:,ind]
    temp_df = temp_df[~temp_df['data'].isna()]
    n_data = temp_df['data'].to_numpy().reshape(len(temp_df['data']),1)
    n_shap = temp_df['shap'].to_numpy().reshape(len(temp_df['shap']),1)
    gam = LinearGAM(s(0,n_splines=5, spline_order = 3)).gridsearch(n_data,n_shap)
    XX = gam.generate_X_grid(term=0, n=500)
    ax.scatter(x_df_wins[feat],shap_values_XGB_test[:,ind],s=1)
    ax.plot(XX, gam.predict(XX), 'r-', alpha = 0.6)
    ax.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--',alpha = 0.3)
#    ax.set_ylim([-0.15,0.15])
    ax.set_title(feat)
    ind+=1
plt.subplots_adjust(hspace=0.3)
plt.savefig('scatter_plots.png')

Below is an analysis where two subgroups are compared to each other. The subgroups are formed using the extreme values of the features.

predictions_df = pd.DataFrame()
f_names = []
max_mean = []
max_pvalues = []
min_mean = []
min_pvalues = []
mean_diff = []
diff_pvalues = []
for feat in bst.feature_names:
    gend_max = x_df_wins[x_df[feat] > x_df_wins[feat].quantile(0.75)]
    gend_min = x_df_wins[x_df[feat] < x_df_wins[feat].quantile(0.25)]
    gend_max_DM = xgboost.DMatrix(gend_max)
    gend_min_DM = xgboost.DMatrix(gend_min)
    pred_max = bst.predict(gend_max_DM)
    pred_min = bst.predict(gend_min_DM)
    _,diff_p_value = ss.ttest_ind(pred_max,pred_min,equal_var=False)
    _,max_p_value = ss.ttest_1samp(pred_max,0)
    _,min_p_value = ss.ttest_1samp(pred_min,0)
    f_names.append(feat)
    max_mean.append(np.mean(pred_max))
    max_pvalues.append(max_p_value)
    min_mean.append(np.mean(pred_min))
    min_pvalues.append(min_p_value)
    mean_diff.append(np.mean(pred_max)-np.mean(pred_min))
    diff_pvalues.append(diff_p_value)
predictions_df['Names'] = f_names
predictions_df['Diff'] = mean_diff
predictions_df['p_value_diff'] = diff_pvalues
predictions_df['Max_mean'] = max_mean
predictions_df['p_value_max'] = max_pvalues
predictions_df['Min_mean'] = min_mean
predictions_df['p_value_min'] = min_pvalues
predictions_df.to_clipboard()

check_normality(pred_min)

plt.hist(pred_max,bins = 20,alpha = 0.5,edgecolor = 'k',color = 'r')
plt.hist(pred_min,bins = 20,alpha = 0.4,edgecolor = 'k')

# Similar analysis for dummy features
gend_true = x_df[x_df['CEO_BOARD_MEMBER'] == 1]
gend_false = x_df[x_df['CEO_BOARD_MEMBER'] == 0]
gend_true_DM = xgboost.DMatrix(gend_true)
gend_false_DM = xgboost.DMatrix(gend_false)
pred_true = bst.predict(gend_true_DM)
pred_false = bst.predict(gend_false_DM)

plt.hist(pred_true,bins = 20,alpha = 0.5,edgecolor = 'k',color = 'r')
plt.hist(pred_false,bins = 20,alpha = 0.4,edgecolor = 'k')



