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

## Reproduce the analysis

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open the notebook:

```bash
jupyter notebook recipe_ratings_analysis.ipynb
```

3. Run all cells in order. The notebook assumes the data files listed below are available.

## Objective

To explore whether simple measurable recipe features can explain differences in user ratings.

## Web App: Recipe Rating Explorer

This project also includes a minimal, single-page web app for exploring which recipe features are associated with higher ratings. The UI is a thin HTML/CSS layer, while all logic and data processing are implemented in Python.

### How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
python -m uvicorn app:app --reload
```

3. Open the app in your browser:

```
http://127.0.0.1:8000
```

### Data requirements (full or sample)

The app expects these files in the repository:

- data/RAW_recipes.csv
- data/RAW_interactions.csv (or archive/interactions_train.csv as fallback)

If the full datasets are not available, the app falls back to the sample files included in this repo:

- data/RAW_recipes_sample.csv
- data/RAW_interactions_sample.csv

### Web app usage

- Search recipes by keyword in the recipe name.
- Filter by minimum rating, preparation time, and ingredient count.
- Click a recipe title to open the detail page with ingredients, steps, and additional metadata.

## Machine Learning Methods

For the May 5 milestone, machine learning methods were applied to predict recipe ratings based on recipe-level features such as preparation time, number of ingredients, calories, and other available numerical variables.

Models used:
- Linear Regression
- Random Forest Regressor

The models were evaluated using train/test split (80/20), test metrics (MAE, RMSE, R² score), and feature importance analysis. Both models achieved low R² scores, indicating that simple recipe features alone have limited predictive power for user ratings.

