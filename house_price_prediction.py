import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
#split data so the model gets to see training data but not validation data
from sklearn.model_selection import train_test_split 

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

#max_leaf_nodes → how complicated the tree can be
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    iowa_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1) 
    iowa_model.fit(train_X, train_y) # TRAIN model with house features X and actual data y
    val_predictions = iowa_model.predict(val_X)  # PREDICTIONS made using X as data 
    val_mae = mean_absolute_error(val_y, val_predictions) #error = actual - predicted
    return(val_mae)


#test different tree sizes to find appropriate one
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {}

for tree_size in candidate_max_leaf_nodes:
    mae = get_mae(tree_size, train_X, val_X, train_y, val_y)
    scores[tree_size] = mae

best_tree_size = min(scores, key=scores.get) #lowest MAE = most accurate
print("Best tree size:", best_tree_size)
print("Validation MAE:", f"{scores[best_tree_size]:,.0f}")
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

#train the final model
final_model.fit(X, y)
