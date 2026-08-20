# Iowa House Price Prediction

I built a beginner machine learning project predicting house prices using
Decision Tree and Random Forest regression models. 

## Project Overview

This project uses the Iowa house price dataset to predict SalePrice
based on features including:

- Lot area
- Year built
- First floor area
- Second floor area
- Number of bathrooms
- Number of bedrooms
- Total rooms

## Models

### Decision Tree Regressor

I first trained a Decision Tree and experimented with different
max_leaf_nodes values to find the tree size with the lowest
validation Mean Absolute Error (MAE).

### Random Forest Regressor

I then improved the previous model by using a RandomForestRegressor and evaluated its performance
using the same validation approach.

## Evaluation

The models were evaluated using Mean Absolute Error (MAE), where lower MAE indicates better prediction accuracy.

## What I Learned

- Pandas and DataFrames
- Features vs prediction targets
- Training and validation data
- Decision Trees
- Random Forests
- Model predictions
- Mean Absolute Error
- Overfitting and underfitting
- Comparing machine learning models

## Acknowledgements

This project was developed with guidance from Kaggle Learn's
Intro to Machine Learning course. To reproduce the
project, download the dataset from Kaggle and place 'train.csv'
in the project directory.
