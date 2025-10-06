'''Plan (pseudocode)

Data: pick a city; collect rentals (CSV). Columns: rent, sqft, latitude, longitude, bedrooms, bathrooms, neighborhood, city.

EDA: check ranges, missing, outliers; plot rent ~ sqft, map lat/lon.

Features:

Spatial: distance to city center, coarse lat/lon buckets.

Size: log(sqft), sqrt(sqft).

Categorical: neighborhood/city one-hot.

Model:

Baselines: Ridge (regularized linear) + RandomForest.

Split train/valid (or time split if dates exist).

RandomizedSearchCV for hyperparams. Metric: RMSE (also MAE, R²).

Pipeline: ColumnTransformer (impute, scale, one-hot) → model. Save with joblib.

CLI: train | predict | eval. Save metrics.json and pipeline.joblib.

Next: add OSM amenity counts, SHAP, dashboard.

Ive put a full, runnable script in the canvas (rent_price_predictor.py) with:

Train/eval/predict subcommands,

Robust preprocessing,

Feature engineering (distance, buckets, transforms),

Model selection (Ridge vs RF) with randomized search,

Saved artifacts + metrics.

Libraries

Core: pandas, numpy, scikit-learn, joblib

Optional (nice-to-have): matplotlib/plotly for EDA, geopandas for maps, xgboost/lightgbm for stronger models, shap for explainability.'''