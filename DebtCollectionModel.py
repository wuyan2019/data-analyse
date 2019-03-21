#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from time import strptime,mktime
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


def LogitRR(x):
    '''
    :param x: 划款率，有的超过1，有的为0.做截断处理
    :return: 将还款率转化成logit变换
    '''
    if x >= 1:
        y = 0.9999
    elif x == 0:
        y = 0.0001
    else:
        y = x
    return np.log(y/(1-y))

def MakeupMissingCategorical(x):
    if str(x) == 'nan':
        return 'Unknown'
    else:
        return x

def MakeupMissingNumerical(x,replacement):
    if np.isnan(x):
        return replacement
    else:
        return x


# In[7]:


mydata = pd.read_csv("prosperLoanData_chargedoff.csv",header = 0)
mydata.head()


# In[8]:


mydata['rec_rate'] = mydata.apply(lambda x: x.LP_NonPrincipalRecoverypayments /(x.AmountDelinquent-x.LP_CollectionFees), axis=1)
# 限定还款率最大为1
mydata['rec_rate'] = mydata['rec_rate'].map(lambda x: min(x,1))


# In[9]:


trainData, testData = train_test_split(mydata,test_size=0.4)


# In[10]:


categoricalFeatures = ['CreditGrade','Term','BorrowerState','Occupation','EmploymentStatus','IsBorrowerHomeowner','CurrentlyInGroup','IncomeVerifiable']

numFeatures = ['BorrowerAPR','BorrowerRate','LenderYield','ProsperRating (numeric)','ProsperScore','ListingCategory (numeric)','EmploymentStatusDuration','CurrentCreditLines',
                'OpenCreditLines','TotalCreditLinespast7years','CreditScoreRangeLower','OpenRevolvingAccounts','OpenRevolvingMonthlyPayment','InquiriesLast6Months','TotalInquiries',
               'CurrentDelinquencies','DelinquenciesLast7Years','PublicRecordsLast10Years','PublicRecordsLast12Months','BankcardUtilization','TradesNeverDelinquent (percentage)',
               'TradesOpenedLast6Months','DebtToIncomeRatio','LoanFirstDefaultedCycleNumber','LoanMonthsSinceOrigination','PercentFunded','Recommendations','InvestmentFromFriendsCount',
               'Investors']


# In[11]:


mydata[numFeatures].describe()


# In[12]:


encodedFeatures = []
encodedDict = {}
for var in categoricalFeatures:
    trainData[var] = trainData[var].map(MakeupMissingCategorical)
    avgTarget = trainData.groupby([var])['rec_rate'].mean()
    avgTarget = avgTarget.to_dict()
    newVar = var + '_encoded'
    newVarSeries = trainData[var].map(avgTarget)
    trainData[newVar] = newVarSeries
    encodedFeatures.append(newVar)
    encodedDict[var] = avgTarget


# In[13]:


trainData[encodedFeatures].head()


# In[14]:


trainData['ProsperRating (numeric)'] = trainData['ProsperRating (numeric)'].map(lambda x: MakeupMissingNumerical(x,0))
trainData['ProsperScore'] = trainData['ProsperScore'].map(lambda x: MakeupMissingNumerical(x,0))

avgDebtToIncomeRatio = np.mean(trainData['DebtToIncomeRatio'])
trainData['DebtToIncomeRatio'] = trainData['DebtToIncomeRatio'].map(lambda x: MakeupMissingNumerical(x,avgDebtToIncomeRatio))


# In[15]:


numFeatures2 = numFeatures + encodedFeatures
X, y= trainData[numFeatures2],trainData['rec_rate']

param_test1 = {'n_estimators':range(10,80,5)}
gsearch1 = GridSearchCV(estimator = RandomForestRegressor(min_samples_split=50,min_samples_leaf=10,max_depth=8,max_features='sqrt' ,random_state=10),
                       param_grid = param_test1, scoring='neg_mean_squared_error',cv=5)
gsearch1.fit(X,y)
gsearch1.best_params_, gsearch1.best_score_
best_n_estimators = gsearch1.best_params_['n_estimators']


# In[16]:


gsearch1.best_params_


# In[17]:


param_test2 = {'max_depth':range(3,21), 'min_samples_split':range(10,100,10)}
gsearch2 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators, min_samples_leaf=10,max_features='sqrt' ,random_state=10,oob_score=True),
                       param_grid = param_test2, scoring='neg_mean_squared_error',cv=5)
gsearch2.fit(X,y)
gsearch2.best_params_, gsearch2.best_score_
best_max_depth = gsearch2.best_params_['max_depth']
best_min_sample_split = gsearch2.best_params_['min_samples_split']

param_test3 = {'min_samples_split':range(50,201,10), 'min_samples_leaf':range(1,20,2)}
gsearch3 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators, max_depth = best_max_depth,max_features='sqrt',random_state=10,oob_score=True),
                       param_grid = param_test3, scoring='neg_mean_squared_error',cv=5)
gsearch3.fit(X,y)
gsearch3.best_params_, gsearch3.best_score_
best_min_samples_leaf = gsearch3.best_params_['min_samples_leaf']
best_min_samples_split = gsearch3.best_params_['min_samples_split']


# In[20]:


numOfFeatures = len(numFeatures2)
mostSelectedFeatures = numOfFeatures/2
param_test4 = {'max_features':range(3,numOfFeatures+1)}
gsearch4 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf,
                                                          min_samples_split=best_min_samples_split,random_state=10,oob_score=True),
                       param_grid = param_test4, scoring='neg_mean_squared_error',cv=5)
gsearch4.fit(X,y)
gsearch4.best_params_, gsearch4.best_score_
best_max_features = gsearch4.best_params_['max_features']


# In[19]:


print(gsearch2.best_params_)
print(gsearch1.best_params_)
print(gsearch3.best_params_)
print(gsearch4.best_params_)


# In[21]:


print('最佳深度: %d' % best_max_depth)
print('最佳深度: %d' % best_n_estimators)
print('最小叶结点样本：%d' % best_min_samples_leaf)
print('最小样本分割： %d' % best_min_samples_split)
print('最大特征: %.2f' % best_max_features)


# In[22]:


cls = RandomForestRegressor(n_estimators=best_n_estimators,
                            max_depth=best_max_depth,
                            min_samples_leaf=best_min_samples_leaf,
                            min_samples_split=best_min_samples_split,
                            max_features=best_max_features,
                            random_state=8,
                            oob_score=True)
cls.fit(X,y)


# In[23]:


trainData['pred'] = cls.predict(trainData[numFeatures2])
trainData['less_rr'] = trainData.apply(lambda x: int(x.pred > x.rec_rate), axis=1)
print(np.mean(trainData['less_rr']))
err = trainData.apply(lambda x: np.abs(x.pred - x.rec_rate), axis=1)
print(np.mean(err))


# In[24]:


# 对测试数据中的字符串变量运用同样的方法进行编码
for var in categoricalFeatures:
    testData[var] = testData[var].map(MakeupMissingCategorical)
    newVar = var + '_encoded'
    testData[newVar] = testData[var].map(encodedDict[var])
    avgnewVar = np.mean(trainData[newVar])
    testData[newVar] = testData[newVar].map(lambda x: MakeupMissingNumerical(x, avgnewVar))

# 对测试数据中的连续变量运用同样的方法进行缺值填补
testData['ProsperRating (numeric)'] = testData['ProsperRating (numeric)'].map(lambda x: MakeupMissingNumerical(x,0))
testData['ProsperScore'] = testData['ProsperScore'].map(lambda x: MakeupMissingNumerical(x,0))
testData['DebtToIncomeRatio'] = testData['DebtToIncomeRatio'].map(lambda x: MakeupMissingNumerical(x,avgDebtToIncomeRatio))

testData['pred'] = cls.predict(testData[numFeatures2])
testData['less_rr'] = testData.apply(lambda x: int(x.pred > x.rec_rate), axis=1)
print(np.mean(testData['less_rr']))
err = testData.apply(lambda x: np.abs(x.pred - x.rec_rate), axis=1)
print(np.mean(err))


# In[ ]:




