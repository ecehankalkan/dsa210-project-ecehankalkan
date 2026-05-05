#!/usr/bin/env python3
import os,sys,warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import statsmodels.api as sm

RND=42

# load data: ratings live in interactions, recipe features live in RAW_recipes
recipes = pd.read_csv('data/RAW_recipes.csv') if os.path.exists('data/RAW_recipes.csv') else None
interactions = None
if os.path.exists('data/RAW_interactions.csv'):
    interactions = pd.read_csv('data/RAW_interactions.csv')
elif os.path.exists('archive/interactions_train.csv'):
    interactions = pd.read_csv('archive/interactions_train.csv')

if recipes is None or interactions is None:
    print('Required input files are missing', file=sys.stderr)
    sys.exit(1)

if 'id' in recipes.columns and 'recipe_id' in interactions.columns:
    df = interactions.merge(recipes, left_on='recipe_id', right_on='id', how='left')
else:
    print('Could not merge recipes and interactions', file=sys.stderr)
    sys.exit(1)

# derive calories from nutrition if needed
if 'calories' not in df.columns and 'nutrition' in df.columns:
    def _extract_calories(value):
        try:
            if pd.isna(value):
                return np.nan
            text = str(value).strip().strip('[]')
            parts = [p.strip() for p in text.split(',')]
            return float(parts[0]) if len(parts) > 0 and parts[0] != '' else np.nan
        except Exception:
            return np.nan
    df['calories'] = df['nutrition'].apply(_extract_calories)

target='rating'
if target not in df.columns:
    print('Target rating not found', file=sys.stderr); sys.exit(1)

candidates = ['minutes','prep_time','num_ingredients','ingredients_count','n_ingredients','calories','fat','protein','carbs','sodium','n_steps','steps_count']
features = [c for c in candidates if c in df.columns]
if len(features)==0:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c!=target]
    features = num_cols[:8]

sub = df[[target]+features].dropna(subset=[target]).copy()
X = sub[features].copy(); y = sub[target].astype(float).copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND)

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
preprocessor = ColumnTransformer([('num', numeric_transformer, features)])

lr_pipe = Pipeline([('pre', preprocessor), ('lr', LinearRegression())])
knn_pipe = Pipeline([('pre', preprocessor), ('knn', KNeighborsRegressor())])
dt_pipe = Pipeline([('pre', preprocessor), ('dt', DecisionTreeRegressor(random_state=RND))])
dummy = Pipeline([('pre', preprocessor), ('d', DummyRegressor(strategy='mean'))])

lr_cv_rmse = -cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
knn_search = GridSearchCV(knn_pipe, {'knn__n_neighbors':[3,5,7,9,11]}, cv=5, scoring='neg_root_mean_squared_error')
knn_search.fit(X_train, y_train)
knn_best = knn_search.best_estimator_; knn_cv_rmse = -knn_search.best_score_
dt_search = GridSearchCV(dt_pipe, {'dt__max_depth':[3,5,7,9,12,None]}, cv=5, scoring='neg_root_mean_squared_error')
dt_search.fit(X_train, y_train)
dt_best = dt_search.best_estimator_; dt_cv_rmse = -dt_search.best_score_
dummy_cv_rmse = -cross_val_score(dummy, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()

models = {'Baseline': dummy, 'Linear': lr_pipe, 'KNN': knn_best, 'DecisionTree': dt_best}
rows = {}; preds = {}
for name,m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test); preds[name]=y_pred
    rows[name] = ( {'Baseline':dummy_cv_rmse,'Linear':lr_cv_rmse,'KNN':knn_cv_rmse,'DecisionTree':dt_cv_rmse}[name],
                   np.sqrt(mean_squared_error(y_test, y_pred)),
                   r2_score(y_test, y_pred) )

# Print table
print('Model              CV RMSE    Test RMSE    Test R^2')
print('--------------------------------------------------')
for k,v in rows.items():
    print(f'{k:17} {v[0]:.4f}    {v[1]:.4f}    {v[2]:.4f}')

best = min(rows.items(), key=lambda x: x[1][1])[0]
cv_best, test_best = rows[best][0], rows[best][1]
print()
print('En iyi model:', best)
print('Neden: en düşük Test RMSE / en yüksek Test R^2')
print('Overfitting:', 'likely' if test_best > cv_best*1.15 else 'not evident')
print()

imp = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X2 = sm.add_constant(X_train_imp); ols = sm.OLS(y_train, X2).fit()

def fmt(var):
    if var is None: return 'not available'
    return ('positive' if float(ols.params[var])>0 else 'negative') + f'  (β = {float(ols.params[var]):.4f}, p = {float(ols.pvalues[var]):.4f})'

prep_var = next((v for v in ['minutes','prep_time'] if v in features), None)
num_var = next((v for v in ['num_ingredients','ingredients_count','n_ingredients'] if v in features), None)
cal_var = next((v for v in ['calories','cal'] if v in features), None)

print('Research answers:')
print('- Preparation time:', fmt(prep_var))
print('- Number of ingredients:', fmt(num_var))
print('- Calories:', fmt(cal_var))
print()

perm = permutation_importance(dt_best, X_test, y_test, n_repeats=20, random_state=RND)
perm_imp = pd.DataFrame({'feature': features, 'importance_mean': perm.importances_mean}).sort_values('importance_mean', ascending=False)

print('Top 5 feature importance:')
for i,row in perm_imp.head(5).reset_index(drop=True).iterrows():
    print(f"{i+1}. {row['feature']} -> {row['importance_mean']:.6f}")

resid = y_test - preds[best]
print()
print('Diagnostics:')
print('Residual mean: {:.6f}, Residual std: {:.6f}'.format(float(resid.mean()), float(resid.std())))
print('Residuals mean ~0:', 'yes' if abs(resid.mean())<1e-6 else 'no')
