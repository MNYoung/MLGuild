# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 19:48:10 2018

@author: michayoung
"""
#import statements
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

#read in the data
data = pd.read_csv('2015_zip.csv')

#read in census education data
cen = pd.read_csv('census_ed_2015.csv')
cen.head()

#extract meaningful columns from census data
edu = cen[['GEO.display-label','HC01_EST_VC02','HC01_EST_VC03','HC01_EST_VC04','HC01_EST_VC05'
          ,'HC01_EST_VC06','HC01_EST_VC08','HC01_EST_VC09','HC01_EST_VC10','HC01_EST_VC11',
          'HC01_EST_VC12','HC01_EST_VC13','HC01_EST_VC14','HC01_EST_VC15']].copy()
edu.head()

#remove text from zipcode and change to numeric
edu['zipcode'] = edu['GEO.display-label'].map(lambda x: x.lstrip('ZCTA5 '))
edu['zipcode'] = pd.to_numeric(edu['zipcode'],errors='coerce')
edu.dtypes

#create new percentage columns for edu
edu['no_high_school'] = (edu['HC01_EST_VC03']+edu['HC01_EST_VC09']+edu['HC01_EST_VC10'])/(edu['HC01_EST_VC02']+edu['HC01_EST_VC08'])
edu['high_school'] = (edu['HC01_EST_VC04']+edu['HC01_EST_VC11'])/(edu['HC01_EST_VC02']+edu['HC01_EST_VC08'])
edu['associates'] = (edu['HC01_EST_VC05']+edu['HC01_EST_VC12']+edu['HC01_EST_VC13'])/(edu['HC01_EST_VC02']+edu['HC01_EST_VC08'])
edu['college'] = (edu['HC01_EST_VC06']+edu['HC01_EST_VC14'])/(edu['HC01_EST_VC02']+edu['HC01_EST_VC08'])
edu['grad'] = edu['HC01_EST_VC15']/edu['HC01_EST_VC08']

#check for null values and replace with 0
edu.isnull().sum()
edu.fillna(0,inplace=True)

#select relevant columns
edu = edu.loc[:,'zipcode':'grad'].copy()

#read in census occupation data
cen1 = pd.read_csv('census_occ_2015.csv')
cen1.head()

#extract meaningful columns from census data
occ = cen1[['GEO.display-label','HC01_EST_VC01','HC01_EST_VC03','HC01_EST_VC06','HC01_EST_VC10',
            'HC01_EST_VC12','HC01_EST_VC15','HC01_EST_VC18','HC01_EST_VC26','HC01_EST_VC29','HC01_EST_VC33']].copy()
occ.head()

#remove text from zipcode and change to numeric
occ['zipcode'] = occ['GEO.display-label'].map(lambda x: x.lstrip('ZCTA5 '))
occ['zipcode'] = pd.to_numeric(occ['zipcode'],errors='coerce')
occ.dtypes

#create new percentage columns for occ
occ['business'] = occ['HC01_EST_VC03']/occ['HC01_EST_VC01']
occ['stem'] = occ['HC01_EST_VC06']/occ['HC01_EST_VC01']
occ['ed_arts'] = (occ['HC01_EST_VC10']-occ['HC01_EST_VC12'])/occ['HC01_EST_VC01']
occ['legal'] = occ['HC01_EST_VC12']/occ['HC01_EST_VC01']
occ['healthcare'] = occ['HC01_EST_VC15']/occ['HC01_EST_VC01']
occ['service'] = occ['HC01_EST_VC18']/occ['HC01_EST_VC01']
occ['sales'] = occ['HC01_EST_VC26']/occ['HC01_EST_VC01']
occ['cons_maint'] = occ['HC01_EST_VC29']/occ['HC01_EST_VC01']
occ['prod_trans'] = occ['HC01_EST_VC33']/occ['HC01_EST_VC01']

#check for null values and replace with 0
occ.isnull().sum()
occ.fillna(0,inplace=True)

#select relevant columns
occ = occ.loc[:,'zipcode':'prod_trans'].copy()

#EDA
data.head()
data.columns
data.shape
data.dtypes
data.describe()
data.info()

#find rows where the number of returns is 0 and drop these rows
data[data['N1']==0].count()
data.drop(data[data.N1 == 0].index, inplace=True)
data.shape
data[data.N1 == 0]

#create refund column
data['refund'] = data['A11902']/data['N11902']

#graphs
plt.scatter(data['A85300'],data['refund'])
plt.show()
plt.scatter(data['N11901'],data['refund'])
plt.show()

'''#categorical variables without zipcode
cat_var = df[['STATE','agi_stub']]
cat_var = cat_var.astype(str)
dummies = pd.get_dummies(cat_var)
dummies.columns
dummies.shape'''

#create percentage columns
df = data.copy()
df['single'] = df['mars1']/df['N1']
df['joint'] = df['MARS2']/df['N1']
df['head_house'] = df['MARS4']/df['N1']
df['prep'] = df['PREP']/df['N1']
df['exemptions'] = df['N2']/df['N1']
df['vita'] = df['TOTAL_VITA']/df['N1']
df['elderly'] = df['ELDERLY']/df['N1']
df['agi_avg'] = df['A00100']/df['N1']
df['total_avg'] = df['A02650']/df['N02650']
df['sal_avg'] = df['A00200']/df['N00200']
df['int_avg'] = df['A00300']/df['N00300']
df['div_avg'] = df['A00600']/df['N00600']
df['qual_div_avg'] = df['A00650']/df['N00650']
df['state_refund'] = df['A00700']/df['N00700']
df['business_income'] = df['A00900']/df['N00900']
df['cap_gain'] = df['A01000']/df['N01000']
df['ret_dist'] = df['A01400']/df['N01400']
df['pensions'] = df['A01700']/df['N01700']
df['unemployment'] = df['A02300']/df['N02300']
df['soc_sec'] = df['A02500']/df['N02500']
df['partnership'] = df['A26270']/df['N26270']
df['stat_adj'] = df['A02900']/df['N02900']
df['educator'] = df['A03220']/df['N03220']
df['self_retire'] = df['A03300']/df['N03300']
df['self_health'] = df['A03270']/df['N03270']
df['ira'] = df['A03150']/df['N03150']
df['stud_loan'] = df['A03210']/df['N03210']
df['tuition'] = df['A03230']/df['N03230']
df['dom_prod'] = df['A03240']/df['N03240']
df['item_ded'] = df['A04470']/df['N04470']
df['state_tax'] =df['A18425']/df['N18425']
df['state_sales_tax'] = df['A18450']/df['N18450']
df['real_estate'] = df['A18500']/df['N18500']
df['taxes_paid'] = df['A18300']/df['N18300']
df['mortgage'] = df['A19300']/df['N19300']
df['contributions'] = df['A19700']/df['N19700']
df['taxable_inc'] = df['A04800']/df['N04800']
df['tax_before_cred'] = df['A05800']/df['N05800']
df['alt_min'] = df['A09600']/df['N09600']
df['excess_prem'] = df['A05780']/df['N05780']
df['tax_cred'] = df['A07100']/df['N07100']
df['for_tax'] = df['A07300']/df['N07300']
df['child_cred'] = df['A07180']/df['N07180']
df['non_ref_ed'] = df['A07230']/df['N07230']
df['ret_cont'] = df['A07240']/df['N07240']
df['child_tax'] = df['A07220']/df['N07220']
df['energy_cred'] = df['A07260']/df['N07260']
df['self_emp_tax'] = df['A09400']/df['N09400']
df['prem_cred'] = df['A85770']/df['N85770']
df['adv_prem_cred'] = df['A85775']/df['N85775']
df['ind_health_pay'] = df['A09750']/df['N09750']
df['tot_tax_pay'] = df['A10600']/df['N10600']
df['eic'] = df['A59660']/df['N59660'] 
df['exc_eic'] = df['A59720']/df['N59720']
df['add_child_cred'] = df['A11070']/df['N11070']
df['ref_ed_cred'] = df['A10960']/df['N10960']
df['net_prem_cred'] = df['A11560']/df['N11560']
df['inc_tax'] = df['A06500']/df['N06500']
df['tax_liab'] = df['A10300']/df['N10300']
df['medicare'] = df['A85530']/df['N85530']
df['inv_tax'] = df['A85300']/df['N85300']
df['tax_due'] = df['A11901']/df['N11901']

#Create variable refund percentage of income
df['refund_percent'] = df['refund']/df['total_avg']
df['refund_percent'].head()

#Create variables for lower bound of refund for AGI class
df['agi_low'] = np.where(df['agi_stub']==1, df['refund'], 
                np.where(df['agi_stub']== 2, df['refund']/25000,
                np.where(df['agi_stub']== 3, df['refund']/50000,
                np.where(df['agi_stub']== 4, df['refund']/75000,
                np.where(df['agi_stub']== 5, df['refund']/100000,
                np.where(df['agi_stub']== 6, df['refund']/200000,1))))))

#Create variables for upper bound of refund for AGI class
df['agi_high'] = np.where(df['agi_stub']==1, df['refund']/24999, 
                np.where(df['agi_stub']== 2, df['refund']/49999,
                np.where(df['agi_stub']== 3, df['refund']/74999,
                np.where(df['agi_stub']== 4, df['refund']/99999,
                np.where(df['agi_stub']== 5, df['refund']/199999,
                np.where(df['agi_stub']== 6, df['refund']/df['total_avg'],1))))))

df['agi_low'].head(10)
df['agi_high'].head(10)
#check for null values and replace with zero
df.isnull().sum()
df.fillna(0,inplace=True)

#combine with census data
df_merged = pd.merge(df,edu, how ='inner', on='zipcode')
df_merged = pd.merge(df_merged,occ, how='inner', on='zipcode')
df_merged.head()
df_merged.columns

edu.head()
#turn agi_stub from numeric into str
df_merged['agi_stub'] = df_merged['agi_stub'].astype(str)
df_merged.dtypes

#total tax paid vs. refund
plt.scatter(df['tot_tax_pay'],df['refund'])
plt.xlabel('Total Tax Payment')
plt.ylabel('Refund Payment')
plt.show()

#create X dataset
X = df_merged.copy()
X.drop(['STATEFIPS','STATE','zipcode','refund','A11902','N11902'],axis=1,inplace=True)

#create dummies for agi_stub
X = pd.get_dummies(X)

#select relevant columns of X
X = X.loc[:,'single':'agi_stub_6']
X.columns
X.shape
X.dtypes
X.head()

#create y dataset
y= df_merged['refund']

'''#check for null values
X.isnull().values.any()
y.isnull().values.any()
y.isnull().sum()

#replace null y values with 0 (was divided by 0)
y.fillna(0,inplace=True)'''

#heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(X.corr(),square=True,cmap='RdBu_r')
plt.show()

'''#standard scalar
data_std = StandardScaler().fit_transform(X)'''

#instantiate linear model
reg = linear_model.LinearRegression()

#cross val score
np.mean(cross_val_score(reg, X, y, cv=5, scoring='neg_mean_squared_error'))

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=52)

#instantiate linear model
reg_all = linear_model.LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
reg_all.score(X_test, y_test)

print('Training accuracy:', reg_all.score(X_train, y_train))
print('Test accuracy:', reg_all.score(X_test, y_test))


y_test.head()
y_pred[:10]

#Random Forest
rf = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)
rf.fit(X_train, y_train)

rf.score(X_train, y_train)
rf.score(X_test, y_test)


# Output feature importance coefficients, map them to their feature name, and sort values
coef = pd.Series(rf.feature_importances_, index = X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()



#Predicted vs. Actual
plt.figure(figsize=(10, 5))
y_pred = rf.predict(X_test)
plt.scatter(y_test, y_pred, s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Refund')
plt.ylabel('Predicted Refund')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout()

'''#visualize decision tree
from sklearn import tree
i_tree = 0
for tree_in_forest in rf.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
        my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
    i_tree = i_tree + 1'''
    
'''from sklearn.tree import export_graphviz
from IPython.display import Image
from graphviz import Source
def tree_viz(tree_object):
    #a function which visualizes a decision tree object
    graph = Source(export_graphviz(tree_object, out_file=None))
    png_bytes = graph.pipe(format='png')
    with open('dtree_pipe.png','wb') as f:
        f.write(png_bytes)
    return Image(png_bytes)

tree_viz(rf.estimators_[0])'''


plt.scatter(df['tot_tax_pay'],df['agi_avg'])
plt.xlabel('Total Tax Payment')
plt.ylabel('Average Adjusted Gross Income')
plt.show()

plt.scatter(df['tot_tax_pay'],df['total_avg'])
plt.xlabel('Total Tax Payment')
plt.ylabel('Total Income')
plt.show()

plt.scatter(df['agi_avg'],df['total_avg'])
plt.xlabel('Average Adjusted Gross Income')
plt.ylabel('Total Income')
plt.show()

plt.scatter(df['taxable_inc'],df['tax_before_cred'])
plt.xlabel('Taxable Income')
plt.ylabel('Income Before Credits')
plt.show()

plt.scatter(df['agi_avg'],df['tax_liab'])
plt.xlabel('Average Adjusted Gross Income')
plt.ylabel('Average Tax Liability')
plt.show()

plt.scatter(df['total_avg'],df['taxable_inc'])
plt.xlabel('Average Total Income')
plt.ylabel('Average Taxable Income')
plt.show()

plt.scatter(df['total_avg'],df['tot_tax_pay'])
plt.xlabel('Average Total Income')
plt.ylabel('Average Tax Payment')
plt.show()

plt.scatter(df['div_avg'],df['qual_div_avg'])
plt.xlabel('Average Ordinary Dividends')
plt.ylabel('Average Qualified Dividends')
plt.show()

df['div_avg'].sum()
df['qual_div_avg'].sum()
#remove highly-correlated features
X['div_combined'] = X['div_avg'] * X['qual_div_avg']

X1 = X.copy()
X1.drop(['agi_avg','tax_before_cred','tax_liab','taxable_inc'],axis=1,inplace=True)

X_norm = preprocessing.normalize(X1)
X1.head()
X_norm
X1.shape
X_norm.shape
X_norm = pd.DataFrame(X_norm,columns=X1.columns)
X_norm.head()

#instantiate random forest without highly-correlated features
X1_train, X1_test, y_train, y_test = train_test_split(X1, y,test_size = 0.3, random_state=52)

rf.fit(X1_train, y_train)

rf.score(X1_train, y_train)
rf.score(X1_test, y_test)

coef = pd.Series(rf.feature_importances_, index = X1.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()

#instantiate random forest with normalized data
X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y,test_size = 0.3, random_state=52)

rf_norm = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)
rf_norm.fit(X_norm_train, y_train)

rf_norm.score(X_norm_train, y_train)
rf_norm.score(X_norm_test, y_test)

coef = pd.Series(rf_norm.feature_importances_, index = X_norm.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()

#divide data by AGI group
X1.drop(['agi_low','agi_high','refund_percent'],axis=1,inplace=True)
y_m = df_merged[['agi_stub','refund']]

X_1 = X1[X1['agi_stub_1']==1].copy()
X_1 = X_1.loc[:,'single':'prod_trans']
y_1 = y_m[y_m['agi_stub']=='1'].copy()
y_1.drop('agi_stub',axis=1,inplace=True) 

X_2 = X1[X1['agi_stub_2']==1].copy()
X_2 = X_2.loc[:,'single':'prod_trans']
y_2 = y_m[y_m['agi_stub']=='2'].copy()
y_2.drop('agi_stub',axis=1,inplace=True)

X_3 = X1[X1['agi_stub_3']==1].copy()
X_3 = X_3.loc[:,'single':'prod_trans']
y_3 = y_m[y_m['agi_stub']=='3'].copy()
y_3.drop('agi_stub',axis=1,inplace=True)

X_4 = X1[X1['agi_stub_4']==1].copy()
X_4 = X_4.loc[:,'single':'prod_trans']
y_4 = y_m[y_m['agi_stub']=='4'].copy()
y_4.drop('agi_stub',axis=1,inplace=True)

X_5 = X1[X1['agi_stub_5']==1].copy()
X_5 = X_5.loc[:,'single':'prod_trans']
y_5 = y_m[y_m['agi_stub']=='5'].copy()
y_5.drop('agi_stub',axis=1,inplace=True)

X_6 = X1[X1['agi_stub_6']==1].copy()
X_6 = X_6.loc[:,'single':'prod_trans']
y_6 = y_m[y_m['agi_stub']=='6'].copy()
y_6.drop('agi_stub',axis=1,inplace=True)

#train test split and fit model to each AGI class
#AGI Class 1
rf1 = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1,test_size = 0.3, random_state=52)

rf1.fit(X_1_train, y_1_train.values.ravel())

rf1.score(X_1_train, y_1_train)
rf1.score(X_1_test, y_1_test)

coef1 = pd.Series(rf1.feature_importances_, index = X_1.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef1.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()
#AGI Class 2
rf2 = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2,test_size = 0.3, random_state=52)

rf2.fit(X_2_train, y_2_train.values.ravel())

rf2.score(X_2_train, y_2_train)
rf2.score(X_2_test, y_2_test)

coef2 = pd.Series(rf2.feature_importances_, index = X_2.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef2.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()
#AGI class 3
rf3 = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)
X_3_train, X_3_test, y_3_train, y_3_test = train_test_split(X_3, y_3,test_size = 0.3, random_state=52)

rf3.fit(X_3_train, y_3_train.values.ravel())

rf3.score(X_3_train, y_3_train)
rf3.score(X_3_test, y_3_test)

coef3 = pd.Series(rf3.feature_importances_, index = X_3.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef3.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()
#AGI class 4
rf4 = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)
X_4_train, X_4_test, y_4_train, y_4_test = train_test_split(X_4, y_4,test_size = 0.3, random_state=52)

rf4.fit(X_4_train, y_4_train.values.ravel())

rf4.score(X_4_train, y_4_train)
rf4.score(X_4_test, y_4_test)

coef4 = pd.Series(rf4.feature_importances_, index = X_4.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef4.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()
#AGI class 5
rf5 = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)
X_5_train, X_5_test, y_5_train, y_5_test = train_test_split(X_5, y_5,test_size = 0.3, random_state=52)

rf5.fit(X_5_train, y_5_train.values.ravel())

rf5.score(X_5_train, y_5_train)
rf5.score(X_5_test, y_5_test)

coef5 = pd.Series(rf5.feature_importances_, index = X_5.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef5.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()
#AGI class 6

rf6 = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)
X_6_train, X_6_test, y_6_train, y_6_test = train_test_split(X_6, y_6,test_size = 0.3, random_state=52)

rf6.fit(X_6_train, y_6_train)

rf6.score(X_6_train, y_6_train)
rf6.score(X_6_test, y_6_test)

coef6 = pd.Series(rf6.feature_importances_, index = X_6.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef6.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()


#Compare AGI to education
df_merged.groupby('agi_stub')
df_merged.columns
plt.scatter(df_merged['agi_stub'],df_merged['college'])
df_merged['college'].head()

#create higher ed field
df_merged['higher_ed'] = df_merged['college'] + df_merged['grad']

import seaborn as sns

sns.boxplot(x='agi_stub',y='college',data=df_merged)
sns.violinplot(x='agi_stub',y='college',data=df_merged)

sns.violinplot(x='agi_stub',y='higher_ed',data=df_merged)

sns.boxplot(x='agi_stub',y='higher_ed',data=df_merged)
sns.boxplot(x='agi_stub',y='no_high_school',data=df_merged)

df_merged['agi_avg'].head()


#create new fields for percent of zip code in each AGI group
df_merged.groupby('agi_stub').higher_ed.mean()
df_merged.groupby('zipcode').agi_avg.mean()

#create column for total population in zipcode
pop_col = df_merged.groupby('zipcode').N1.sum().reset_index(name='total_pop')
df_merged = pd.merge(df_merged, pop_col, on=['zipcode'])

#create column for percentage of people in the zipcode in each AGI class 
df_merged['agi_percent'] = df_merged['N1']/df_merged['total_pop']

#new dataframe for Edu vs AGI class
ed_agi = df_merged.copy()

#drop unnecessary columns
ed_agi.drop(['STATEFIPS','STATE'],axis=1,inplace=True)
ed_agi.drop(ed_agi.loc[:,'mars1':'A11902'],axis=1,inplace=True)

ed_agi['agi_stub'] = pd.to_numeric(ed_agi['agi_stub'],errors='coerce')
ed_agi.head()
ed_agi.dtypes
#total population in each AGI class
df_merged.groupby('agi_stub').N1.sum()

#graph AGI percentage vs Higher Education for each AGI group
for agi in range(1,7):
    plt.scatter(x='agi_percent',y='higher_ed',data=ed_agi[ed_agi['agi_stub']==agi],alpha=.2)
    plt.title('AGI %s' %agi)
    plt.xlabel('AGI Percentage')
    plt.ylabel('Higher Education')
    plt.show()

ed_agi[['zipcode','agi_avg','agi_stub']].sort_values(by='agi_avg',ascending=False).head()

plt.figure(figsize=(10, 10))
sns.heatmap(ed_agi.corr(),square=True,cmap='RdBu_r')
plt.show()

ed_agi[ed_agi['zipcode']==72712]
df[df['zipcode']==72712]
ed_agi.sort_values(by='tax_liab',ascending=False).head()
ed_agi.sort_values(by='refund',ascending=False).head()
ed_agi.sort_values(by='tax_due',ascending=False).head()
ed_agi.sort_values(by='higher_ed',ascending=False).head()
df.sort_values(by='A00100',ascending=False).head()


df_merged.to_csv('tax_census.csv')
Xc = ed_agi.drop('agi_stub',axis=1)
yc = ed_agi['agi_stub']

from sklearn.ensemble import RandomForestClassifier

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc,yc,test_size=0.3)

# Instantiate a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 100,n_jobs=-1, random_state=9)

# Fit the Random Forest Classifier
rfc.fit(Xc_train,yc_train)


rfc.score(Xc_train,yc_train)
rfc.score(Xc_test,yc_test)

coefc = pd.Series(rfc.feature_importances_, index = Xc_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coefc.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()

#predict percentage of people in each AGI group
y_agi = df_merged['agi_percent']
X_agi = ed_agi.drop('total_pop',axis=1)

Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_agi,y_agi,test_size=0.3)

rf_agi = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)

rf_agi.fit(Xa_train,ya_train)

rf_agi.score(Xa_train,ya_train)
rf_agi.score(Xa_test,ya_test)

coefa = pd.Series(rf_agi.feature_importances_, index = Xa_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coefa.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()

#predicted vs actual graph
plt.figure(figsize=(10, 5))
y_pred = rf_agi.predict(Xa_test)
plt.scatter(ya_test, y_pred, s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual AGI Percent')
plt.ylabel('Predicted AGI Percent')

plt.plot([min(ya_test), max(ya_test)], [min(ya_test), max(ya_test)])
plt.tight_layout()

#Predict higher education percentage
y_ed = df_merged['higher_ed']
X_ed = ed_agi.drop(['higher_ed', 'college', 'grad','high_school','no_high_school','associates'],axis=1)

Xe_train, Xe_test, ye_train, ye_test = train_test_split(X_ed,y_ed,test_size=0.3)

rf_ed = RandomForestRegressor(n_estimators = 100,n_jobs=-1, random_state=9)

rf_ed.fit(Xe_train,ye_train)

rf_ed.score(Xe_train,ye_train)
rf_ed.score(Xe_test,ye_test)

coefe = pd.Series(rf_ed.feature_importances_, index = Xe_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coefe.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()

#Read in 2012-2014 data
df2012 = pd.read_csv('2012_zip.csv')
df2013 = pd.read_csv('2013_zip.csv')
df2014 = pd.read_csv('2014_zip.csv')

df2012.shape
df2013.shape
df2014.shape
data.shape
df2012.columns

#create new columns with the year of the data
df2012['year'] = 2012
df2013['year'] = 2013
df2014['year'] = 2014
data['year'] = 2015

#find columns present in 2015 that are not present in 2012
columns_to_drop = data.columns.difference(df2012.columns)

#append data
df_years = df2014.append(data)
df_years = df_years.append(df2013)
df_years = df2012.append(df_years)
df_years.head()
df_years.columns
#drop columns for which we do not have data in 2012
df_years.drop(columns_to_drop,axis=1,inplace=True)
df_years.columns.difference(df2012.columns)
df_years.drop(['A85330','N85330'],axis=1,inplace=True)

df_years.groupby('year').N59660.sum()
df_years.groupby('year').A59660.sum()


plt.scatter(df_years['year'],df_years['N59660'].sum())
plt.xlabel('Year')
plt.ylabel('Number of returns with earned income credit')
plt.show()

#create new data frame grouped by year
year = df_years.groupby([(df_years.year)]).sum()

#EIC Graphs
plt.plot(year.index,year.N59660)
plt.title('Returns with Earned Income Credit')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,year.A59660)
plt.title('Total Earned Income Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,(year.A59660/year.N59660))
plt.title('Average Earned Income Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

#Child Tax Credit Graphs
plt.plot(year.index,year.N07220)
plt.title('Returns with Child Tax Credits')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,year.A07220)
plt.title('Total Child Tax Credits Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,(year.A07220/year.N07220))
plt.title('Average Child Tax Credits Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

#Additional child tax credit graphs
plt.plot(year.index,year.N11070)
plt.title('Returns with Additional Child Tax Credits')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,year.A11070)
plt.title('Total Additional Child Tax Credits Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,(year.A11070/year.N11070))
plt.title('Average Additional Child Tax Credits Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

#Residential Energy Credit Graphs
plt.plot(year.index,year.N07260)
plt.title('Returns with Residential Energy Credit')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,year.A07260)
plt.title('Total Residential Energy Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,(year.A07260/year.N07260))
plt.title('Average Residential Energy Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

#Child & Dependent Care Credit Graphs
plt.plot(year.index,year.N07180)
plt.title('Returns with Child & Dependent Care Credit')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,year.A07180)
plt.title('Total Child & Dependent Care Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,(year.A07180/year.N07180))
plt.title('Average Child & Dependent Care Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

#Total Tax Credit Graphs
plt.plot(year.index,year.N07100)
plt.title('Returns with Total Tax Credit')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,year.A07100)
plt.title('Total Tax Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

plt.plot(year.index,(year.A07100/year.N07100))
plt.title('Average Tax Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2012,2016,1))
plt.show()

#Read in 2006-2011 data
df2006 = pd.read_csv('2006_zip.csv')
df2007 = pd.read_csv('2007_zip.csv')
df2008 = pd.read_csv('2008_zip.csv')
df2009 = pd.read_csv('2009_zip.csv')
df2010 = pd.read_csv('2010_zip.csv')
df2011 = pd.read_csv('2011_zip.csv')

#check shape of the data
df2006.shape
df2007.shape
df2008.shape
df2009.shape
df2010.shape
df2011.shape

#check columns of the data
df2012.columns
df2011.columns
df2010.columns
df2009.columns

#rename zipcode column for 2011
df2011 = df2011.rename(columns={'ZIPCODE': 'zipcode'})

#check if the data contains the Residential Energy Credit field
df2009.N07260.head()
df2008.NO7260.head()

#create new columns with the year of the data
df2011['year'] = 2011
df2010['year'] = 2010
df2009['year'] = 2009

#find columns present in 2015 that are not present in 2009
columns_to_drop1 = df_years.columns.difference(df2009.columns)

#append data
df_years1 = df2011.append(df_years)
df_years1 = df_years1.append(df2010)
df_years1 = df_years1.append(df2009)
df_years1.head()
df_years1.columns

#drop columns for which we do not have data in 2012
df_years1.drop(columns_to_drop1,axis=1,inplace=True)

#create new data frame grouped by year
year1 = df_years1.groupby([(df_years1.year)]).sum()

#EIC Graphs
plt.plot(year1.index,year1.N59660)
plt.title('Returns with Earned Income Credit')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,year1.A59660)
plt.title('Total Earned Income Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,(year1.A59660/year1.N59660))
plt.title('Average Earned Income Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

#Child Tax Credit Graphs
plt.plot(year1.index,year1.N07220)
plt.title('Returns with Child Tax Credits')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,year1.A07220)
plt.title('Total Child Tax Credits Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,(year1.A07220/year1.N07220))
plt.title('Average Child Tax Credits Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

#Additional child tax credit graphs
plt.plot(year1.index,year1.N11070)
plt.title('Returns with Additional Child Tax Credits')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,year1.A11070)
plt.title('Total Additional Child Tax Credits Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,(year1.A11070/year1.N11070))
plt.title('Average Additional Child Tax Credits Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

#Residential Energy Credit Graphs
plt.plot(year1.index,year1.N07260)
plt.title('Returns with Residential Energy Credit')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,year1.A07260)
plt.title('Total Residential Energy Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,(year1.A07260/year1.N07260))
plt.title('Average Residential Energy Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

#Child & Dependent Care Credit Graphs
plt.plot(year1.index,year1.N07180)
plt.title('Returns with Child & Dependent Care Credit')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,year1.A07180)
plt.title('Total Child & Dependent Care Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,(year1.A07180/year1.N07180))
plt.title('Average Child & Dependent Care Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

#Total Tax Credit Graphs
plt.plot(year1.index,year1.N07100)
plt.title('Returns with Total Tax Credit')
plt.xlabel('Year')
plt.ylabel('Number of Returns')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,year1.A07100)
plt.title('Total Tax Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()

plt.plot(year1.index,(year1.A07100/year1.N07100))
plt.title('Average Tax Credit Amount')
plt.xlabel('Year')
plt.ylabel('Thousands of Dollars')
plt.xticks(np.arange(2009,2016,1))
plt.show()