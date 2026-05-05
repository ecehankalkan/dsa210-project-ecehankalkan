# dsa210-project-ecehankalkan
# DSA210 Project – Recipe Ratings Analysis

This project investigates how recipe characteristics such as preparation time, calorie content, and number of ingredients relate to user ratings.

## Data Sources

* Food.com recipe dataset
* External nutrition dataset

## Project Scope

The analysis includes:

* data loading and merging
* data cleaning and preprocessing
* feature engineering
* exploratory data analysis (EDA)
* hypothesis testing with t-tests
* machine learning models for predicting average recipe ratings

## Main File

* `recipe_ratings_analysis.ipynb`

## Objective

To explore whether simple measurable recipe features can explain differences in user ratings.

## Machine Learning Methods

For the May 5 milestone, machine learning methods were applied to predict recipe ratings based on recipe-level features such as preparation time, number of ingredients, calories, and other available numerical variables.

Models used:
- Linear Regression
- Random Forest Regressor

The models were evaluated using train/test split (80/20), test metrics (MAE, RMSE, R² score), and feature importance analysis. Both models achieved low R² scores, indicating that simple recipe features alone have limited predictive power for user ratings.

