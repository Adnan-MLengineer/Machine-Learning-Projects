# Importing the Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

a = pd.read_csv("Real estate.csv")

house_price_dataset = a

print(house_price_dataset)

house_price_dataframe = house_price_dataset

house_price_dataframe.head()

house_price_dataframe['price'] = house_price_dataset.target

house_price_dataframe.head()

house_price_dataframe.shape

house_price_dataframe.isnull().sum()

house_price_dataframe.describe()

# Understanding the corelation between various features in Dataset
"""
Add blockquote

1. Positive Correlation
2. Negative Correlation
"""

house_price_dataframe_numeric = house_price_dataframe.select_dtypes(include=[np.number])
correlation = house_price_dataframe_numeric.corr()
print(correlation)

# Constructing a heatmap to understand the correaltion
plt.figure(figsize=(10,10), dpi=100)
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Greens')
plt.shows()

X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

"""# **Model Training: XGBoost Regressor**"""

# loading the model
model = XGBRegressor()

# training the model with X_train
model.fit(X_train, Y_train)

"""# **Evaluation: Prediction on Training Data**"""

# Accuracy on Prediction on Training Data
training_data_prediction = model.predict(X_train)

print(training_data_prediction)

# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)
# Mean Abosulte Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
print("R squared error : ", score_1)
print("Mean Absolute Error : ", score_2)

"""# **Visualizing the Actual Prices and Predicted Prices**"""

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

"""# **Prediction on Test Data**"""

# Accuracy on Prediction on Training Data
test_data_prediction = model.predict(X_test)

# R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)
# Mean Abosulte Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("R squared error : ", score_1)
print("Mean Absolute Error : ", score_2)
