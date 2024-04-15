import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fire_data = pd.read_csv(r"E:\DATA IS GOOD\INTERNSHIP ASSINGMENT\Internship 1205\Model Deployment Project on Local Host - 11\Forest Fire Project\Forest_fire.csv")
fire_data.head()

fire_data.shape
fire_data.isna().sum()

import seaborn as sns
corr_matrix = fire_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")

fig, axs = plt.subplots(6, 1, figsize=(8, 8))
for i, col in enumerate(['Oxygen', 'Temperature', 'Humidity']):
    axs[i].scatter(fire_data[col], fire_data['Fire Occurrence'])
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('Fire Occurrence')
plt.tight_layout()
plt.show()
pd.plotting.scatter_matrix(fire_data, figsize=(6, 6))
plt.show()

# As per above data their no outlier points present 
train = fire_data.iloc[: , 1:-1]
test  = fire_data.iloc[: , -1]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(train, test)

# Using pickel
import pickle
pickle.dump(lr, open('model.pkl', 'wb'))

forest_model_pickle = pickle.load(open('model.pkl', 'rb'))

# Checking mode performance
print(forest_model_pickle.predict([[40,50,10]]))
