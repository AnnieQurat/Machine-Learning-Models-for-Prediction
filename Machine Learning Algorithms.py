#!/usr/bin/env python
# coding: utf-8

# In[18]:


#import libraries
import numpy as np
import scipy as scipy
from scipy import misc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')


# In[19]:


#load the data
train = pd.read_csv('training_A3.csv')
test = pd.read_csv('test_A3.csv')


# ## DATA PREPARATION 

# In[20]:


#split train dataset in x (independent) and y (dependent)
train_x_a = train.drop("price",axis=1)
train_y = train[["price"]]

train_x_a.head()


# In[21]:


#view null rows in price column
train_y[pd.isnull(train_y).any(axis = 1)]


# In[22]:


##Delete rows that dependent has null value
train_x_a = train_x_a.drop([18,37,47,69,71,81,82,100,107])
train_x_a.head(2)


# In[23]:


#find columns containing missing (NaN) values, store column names in a list called nan_columns:
train_x_a.isnull().sum()


# In[24]:


##fill missing values of nan column with median
nan_train_columns = train_x_a[['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg']]
train_x_b = nan_train_columns.fillna(nan_train_columns.median())
median_values = nan_train_columns.median()
#drop null values from train_y
train_y = train_y.dropna(subset = ['price'])


# In[25]:


##columns that contain categorical values
categorical_columns = (train_x_a.loc[:, train_x_a.dtypes == object])
train_x_c = pd.Categorical(categorical_columns)
train_x_c = pd.get_dummies(categorical_columns)
train_x_d = pd.concat([train_x_b, train_x_c], axis = 1)


# In[26]:


##Feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_x_e = ss.fit_transform(train_x_d)
train_x_e = pd.DataFrame(train_x_e, columns = train_x_d.columns)


# In[27]:


test_x = pd.DataFrame(test,  columns = ['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','wheelbase','carlength','carwidth','carheight','curbweight','enginetype','cylindernumber','enginesize','fuelsystem','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg'])
test_y = pd.DataFrame(test,  columns = ['price'])

test_y[pd.isnull(test_y).any(axis = 1)]


# In[28]:


test_x = test_x.drop([10])
test_x.head()


# In[29]:


test_x.isnull().sum()


# In[30]:


##Fill missing value of nan column with median values
nancols = test_x[['carwidth','wheelbase','carheight','carlength','stroke','curbweight','enginesize','compressionratio','horsepower','peakrpm','citympg','highwaympg']]
test_x_a = nancols.fillna(median_values)

test_x_a = pd.concat([test_x['boreratio'], test_x_a ], axis = 1)
test_y = test_y.dropna(subset = ['price'])
test_x_a.head()


# In[31]:


##Columns that contain categorial values
categorical_columns= (test_x.loc[:, test_x.dtypes == object])
categorical_columns.head()


# In[32]:


##Encode the categorial values
test_x_b = pd.Categorical(categorical_columns)
test_x_b = pd.get_dummies(categorical_columns)
test_x_c = pd.concat([test_x_a, test_x_b], axis = 1)
test_x_c.head()


# In[33]:


####Feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
test_x_d = ss.fit_transform(test_x_c)
test_x = pd.DataFrame(test_x_d, columns = test_x_c.columns)
test_x.head(2)


# ## KNN REGGRESSOR TO PREDICT CAR PRICES 

# ### Training Vs Validation Plot:

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors


# In[18]:


# Split dataset into a validation and new training set 
from sklearn.model_selection import train_test_split

train_split_x, val_split_x, train_split_y, val_split_y = train_test_split(train_x_e, train_y, test_size=0.2, random_state=0)


# In[19]:


# For every integer value k between 1 and 100 create and record a KNN Regression model’s training and validation MSEs
# where the KNN model’s number of neighbours is k.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn_training_errors = []
knn_validation_error = []
for k in range(1, 101):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_split_x, train_split_y)
    train_pred = knn.predict(train_split_x)
    val_pred = knn.predict(val_split_x)
    train_error = mean_squared_error(train_split_y, train_pred)
    val_error = mean_squared_error(val_split_y, val_pred)
    knn_training_errors.append(train_error)
    knn_validation_error.append(val_error)
    


# In[20]:


# training vs validation plot
plt.plot(knn_training_errors, 'r-', label = 'train')
plt.plot(knn_validation_error, 'b-', label = 'validation')
plt.legend(loc='upper right', fontsize = 14)
plt.xlabel('K value')
plt.ylabel('MSE')


# ### Test the model:

# In[21]:


neighbors = list(range(1,30,1))
optimal_k = neighbors[knn_validation_error.index(min(knn_validation_error))]
optimal = optimal_k +1
print('the best k is : ', optimal)


# In[22]:


best_knn = KNeighborsRegressor(n_neighbors= 3)
best_knn.fit(train_split_x, train_split_y)


# In[23]:


knn_preds = best_knn.predict(val_split_x)
knn_mse = mean_squared_error(val_split_y, knn_preds)
print(knn_mse)


# ## DECISION TREE REGRESSOR TO PREDICT CAR PRICES

# ### Grid Search to find best model:

# In[24]:


#import libraries
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[25]:


# Create a regressor object
regressor = DecisionTreeRegressor(random_state=0)


# In[26]:


# Define the grid of hyperparameters
params = {"max_depth": [None,1,5,10], 
          "max_features": ["auto", "sqrt", "log2"], 
          "min_samples_split": [0.01, 0.05, 0.1, 0.3], 
          "max_leaf_nodes": [10, 50, 100, 250]}


# In[27]:


# Instantiate a 5-fold CV grid search object 'decision_tree_grid'
decision_tree_grid = GridSearchCV(regressor, params, cv=5,scoring="neg_mean_squared_error")


# In[28]:


# Fit the model to the grid
decision_tree_grid.fit(train_x_e,train_y)


# In[29]:


# Find best hyperparametrs
best_acc = decision_tree_grid.best_score_
print(best_acc)

best_parameters = decision_tree_grid.best_params_
print(best_parameters)


# ### Test the model:

# In[30]:


# Fit the best hyperparameters to the tree
best_tree = DecisionTreeRegressor(max_leaf_nodes = 50, min_samples_split = 0.1, max_depth = 5, max_features = 'auto')
best_tree.fit(train_x_e, train_y)


# In[31]:


test_x.head()


# In[32]:


tree_preds = best_tree.predict(test_x)


# In[33]:


tree_mse = mean_squared_error(test_y, tree_preds)
print('MSE: ', tree_mse )


# ## ENSEMBLE BAGGING REGRESSOR TO PREDICT CAR PRICES

# ### Grid Search to find best model:

# In[34]:


# Import libraries
from sklearn.ensemble import BaggingRegressor


# In[35]:


from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


bagging_res = BaggingRegressor(random_state=0)

params_bagging ={'base_estimator': [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(random_state = 0)],
        'n_estimators': [25, 50, 100, 250 ],
        'bootstrap_features': [False, True],
        'random_state': [0]}

bagging_grid = GridSearchCV(bagging_res, params_bagging, cv = 5, scoring = 'neg_mean_squared_error')
br_fit= bagging_grid.fit(train_x_e, train_y)

bagging_score = bagging_grid.score(test_x, test_y)

print('Best Parameters: ', br_fit.best_params_)

print('MSE: ', np.abs(bagging_score))


# ### Test the model:

# In[42]:


best_bag = BaggingRegressor(base_estimator = DecisionTreeRegressor(random_state = 0), bootstrap_features = True, n_estimators = 25, random_state = 0)
best_bag.fit(train_x_e, train_y)
bagging_preds = best_bag.predict(test_x)


# In[43]:


bag_mse = mean_squared_error(test_y, bagging_preds)
print('MSE: ', bag_mse )


# ## RANDOM FOREST REGRESSOR TO PREDICT CAR PRICES

# ### Grid Search to find best model:

# In[54]:


from sklearn.ensemble import RandomForestRegressor


# In[60]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestRegressor
classifier_random =RandomForestRegressor(random_state = 0)
params ={'n_estimators': [25, 50, 100, 250 ],
         "max_depth":[None,1,5,10]}
grid_random=GridSearchCV(classifier_random,params,cv=5,scoring="neg_mean_squared_error")
grid_random.fit(train_x_e,train_y)


# In[61]:


# Fit it to the model
random_forest_grid.fit(train_x_e,train_y)


# In[62]:


# Find the best hyperparameters
best_acc=grid_random.score(test_x, test_y)
print(best_acc)

best_parameters=grid_random.best_params_
print(best_parameters)
print('MSE: ', np.abs(best_acc))


# ### Test the model:

# In[65]:


best_random_forest= RandomForestRegressor(max_depth = 10, n_estimators = 250)
best_random_forest.fit(train_x_e, train_y)


# In[66]:


random_forest_preds = best_random_forest.predict(test_x)


# In[67]:


random_forest_mse = mean_squared_error(test_y, random_forest_preds)
print('MSE: ', random_forest_mse)


# ### Feature Importances:

# In[52]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[53]:


top_3_rf_features = []
print(top_3_rf_features)


# In[ ]:




