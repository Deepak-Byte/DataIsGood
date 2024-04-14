#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


car_data = pd.read_csv(r"C:\\Users\\DEEPAK\\Downloads\\Car Dataset.csv")
car_data.head()


# In[4]:


car_data.shape


# # Data Preprocessing

# In[5]:


car_data.info()


# In[6]:


car_data.isna().sum()


# In[7]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in car_data:
    if car_data[col].dtype == 'object':
        car_data[col] = label_encoder.fit_transform(car_data[col])
car_data.head()


# In[8]:


car_data['owner'].value_counts()


# In[10]:


import matplotlib.pyplot as plt
fig, axs = plt.subplots(6, 1, figsize=(8, 12))
for i, col in enumerate(['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']):
    axs[i].scatter(car_data[col], car_data['selling_price'])
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('selling_price')
plt.tight_layout()
plt.show()
pd.plotting.scatter_matrix(car_data, figsize=(10, 10))
plt.show()


# In[11]:


Q1 = car_data['year'].quantile(0.25)
Q3 = car_data['year'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(lower_bound, upper_bound)
outliers = ((car_data['year'] < lower_bound) | (car_data['year'] > upper_bound))
outliers.value_counts()


# In[12]:


Q1 = car_data['km_driven'].quantile(0.25)
Q3 = car_data['km_driven'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(lower_bound, upper_bound)
print(Q1,Q3)
outliers = ((car_data['km_driven'] < lower_bound) | (car_data['km_driven'] > upper_bound))
outliers.value_counts()


# In[13]:


# data_no_outliers = car_data[~outliers.any(axis=1)]

# # Print the shape of the original and filtered data to see the number of removed outliers
# print("Original data shape:", car_data.shape)

car_data_new = car_data[(car_data['year'] > 2004) & (car_data['year'] < 2023) & (car_data['km_driven'] > -47500) & (car_data['km_driven'] < 172500)]
car_data_new.head()


# In[14]:


X = car_data_new.drop(columns = 'selling_price', axis = 1)
Y = car_data_new['selling_price']


# In[15]:


array = Y.values
Y_new = array.reshape(-1, 1)


# In[17]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y_new)

# Normalization
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X_new)


# # Feature Engineering & Feature Selection

# In[18]:


# For feature selection we use RFECV (Recursive feature elimination and cross validation)
import seaborn as sns
fig, ax = plt.subplots(figsize = (10,10))
corr = car_data.corr()
sns.heatmap(corr, annot = True, ax=ax)


# In[19]:


Y_scaled.shape, X_scaled.shape


# In[20]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, Y_scaled, test_size = 0.2, random_state = 42)
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


# In[21]:


# Checking data weather it is standardize or not, for same mean should be around 1
plt.ylim(-1,1)
means = []
for i in range (X_scaled.shape[1]):
    means.append(np.mean(xtest[:,i]))
plt.plot(means, scaley = False) 


# # Linear regression

# In[22]:


from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
model = LinearRegression()
rfecv = RFECV(model, step=1, min_features_to_select = 15, n_jobs = -1 )
rfecv.fit(xtrain, ytrain)


# In[23]:


# Get the selected features
selected_features = rfecv.support_


# In[24]:


# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross-validation score (R-squared)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()


# In[25]:


X_train_selected = rfecv.transform(xtrain)
X_test_selected = rfecv.transform(xtest)


# In[26]:


estimator_selected = LinearRegression()
estimator_selected.fit(X_train_selected, ytrain)


# In[27]:


y_pred = estimator_selected.predict(X_test_selected)


# In[28]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(ytest, y_pred)
r2 = r2_score(ytest, y_pred)


# In[29]:


print(" MSE :- ", mse , " \n R2 :- ", r2)


# # Support Vector Regression

# In[30]:


from sklearn.svm import SVR


# In[31]:


svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
svr.fit(xtrain, ytrain)


# In[33]:


y_pred_svm = svr.predict(xtest)


# In[36]:


from sklearn.metrics import mean_squared_error, r2_score
mse_svm = mean_squared_error(ytest, y_pred_svm)
r2_svm = r2_score(ytest, y_pred_svm)


# In[37]:


print(" MSE :- ", mse_svm , " \n R2 :- ", r2_svm)


# # Gradient Boosting 

# In[40]:


from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2, 'learning_rate': 0.1, 'loss': 'squared_error'}
gb_regressor = GradientBoostingRegressor(**params)
gb_regressor.fit(xtrain, ytrain)


# In[42]:


y_pred_gb = gb_regressor.predict(xtest)


# In[45]:


from sklearn.metrics import mean_squared_error, r2_score
mse_gb = mean_squared_error(ytest, y_pred_gb)
r2_gb = r2_score(ytest, y_pred_gb)


# In[47]:


print(" MSE :- ", mse_gb , " \n R2 :- ", r2_gb)


# # Random Forest Regression

# In[49]:


from sklearn.ensemble import RandomForestRegressor


# In[51]:


params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2,
          'min_samples_leaf': 1, 'random_state': 42}
rf_regressor = RandomForestRegressor(**params)
rf_regressor.fit(xtrain, ytrain)


# In[52]:


y_pred_rfr = rf_regressor.predict(xtest)


# In[55]:


from sklearn.metrics import mean_squared_error, r2_score
mse_rfr = mean_squared_error(ytest, y_pred_rfr)
r2_rfr = r2_score(ytest, y_pred_rfr)


# In[56]:


print(" MSE :- ", mse_rfr , " \n R2 :- ", r2_rfr)

