import argparse
import joblib
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
from scipy.stats import randint, uniform

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_score

# Model
from sklearn.ensemble import RandomForestRegressor


def train(csv_fn, model_fn):
    df = pd.read_csv(csv_fn)

    df["malaria_incidence"] = df["cases"]/df["population"] * 10000 #cases per 10000 
    
    df = df.sort_values(["location", "date"])
    for var in ["mean_temperature", "rainfall", "cases", "malaria_incidence"]:
        for lag in [1, 2, 3]:  # 1-3 months lag
            df[f"{var}_lag{lag}"] = df.groupby("location")[var].shift(lag)

    # Log-transform incidence (avoid skew)
    df["malaria_incidence_log"] = np.log1p(df["malaria_incidence"])

    # Drop rows with missing lags (first 1–3 months per location)
    df_ml = df.dropna().reset_index(drop=True)

    # Define features and target
    features = [
        "rainfall","mean_temperature",
        "rainfall_lag1", "rainfall_lag2", "rainfall_lag3",
        "mean_temperature_lag1","mean_temperature_lag2","mean_temperature_lag3",
        "malaria_incidence_lag1","malaria_incidence_lag2","malaria_incidence_lag3"
    ]

    X = df_ml[features]
    y = df_ml["malaria_incidence_log"]

    # we’ll group by location to avoid spatial leakage
    groups = df_ml["location"]

    # Custom scorers (use RMSE as primary)
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    rmse_scorer = make_scorer(rmse, greater_is_better=False)  # negative for sklearn

    # Random Forest + search space
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": randint(100, 1001),           # 100–1000 trees
        "max_depth": randint(6, 41),                  # 6–40 (None can overfit; try explicit depths)
        "min_samples_split": randint(2, 21),          # 2–20
        "min_samples_leaf": randint(1, 11),           # 1–10
        "max_features": ["sqrt", "log2", 0.5, 0.7],   # common good options
        "bootstrap": [True]                           # RF default & usually best
    }

    # GroupKFold prevents training & testing on the same location
    cv = GroupKFold(n_splits=5)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=60,                 # increase if you want a deeper search
        scoring=rmse_scorer,       # primary metric: (negative) RMSE on log scale
        refit=True,                # refit best on full data
        cv=cv.split(X, y, groups=groups),
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )

    search.fit(X, y)

    print("Best params:", search.best_params_)
    print("Best CV RMSE (log scale):", -search.best_score_)

    best_rf = search.best_estimator_

    # Fit final model on all data for downstream use
    best_rf.fit(X, y)

    #_--------------------------------------
    #Now have the best model fit with tuned hyperparameters, not quite sure how to predict from here
    #Are all the regions trained together? Is the location used to split nodes in the trees?



    # split df into one df per distinct location
    models = {}
    for district in df['location'].unique():
        print("Training for district: ", district)
        district_df = df[df['location'] == district]
        model = train_for_district(district_df)
        models[district] = model
    
        model_file_name = "model_" + district + ".bin"
        joblib.dump(model, model_file_name)
    
    joblib.dump(model, model_fn)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a minimalist forecasting model.')

    parser.add_argument('csv_fn', type=str, help='Path to the CSV file containing input data.')
    parser.add_argument('model_fn', type=str, help='Path to save the trained model.')
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn)


