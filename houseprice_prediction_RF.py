import pandas as pd

from sklearn.metrics import mean_absolute_error
#split data so the model gets to see training data but not validation data
from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestRegressor

iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
print(home_data.describe()) #prints summary statistics
home_data = home_data.dropna(axis=0) #drops missing values, think of 'na' as not available
print(home_data.columns) #prints the column names as a list
y = home_data.SalePrice 
feature_names = ['LotArea',
'YearBuilt',
'1stFlrSF',
'2ndFlrSF',
'FullBath',
'BedroomAbvGr',
'TotRmsAbvGrd'] 
X = home_data[feature_names]

#Target(y) is what you're trying to predict and Feature(X) is the data given to make the prediction

train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

#train a Random Forest model for comparison with the Decision Tree
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
