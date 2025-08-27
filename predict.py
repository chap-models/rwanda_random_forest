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

def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    # get all unique districts from historic data
    future_df = pd.read_csv(future_climatedata_fn)
    historic_df = pd.read_csv(historic_data_fn)
    historic_df["malaria_incidence"] = historic_df["disease_cases"]/historic_df["population"] * 10000 #cases per 10000 

    full_df = pd.concat([historic_df, future_df], ignore_index=True)

    full_df["date"] = pd.to_datetime(full_df["time_period"], format="%Y-%m")

    future_df["date"] = pd.to_datetime(future_df["time_period"], format="%Y-%m")

    # Find the earliest future timepoint
    earliest_time = future_df["date"].min()

    full_df = full_df.sort_values(["location", "date"])
    for var in ["mean_temperature", "rainfall", "disease_cases", "malaria_incidence"]:
        for lag in [1, 2, 3]:  # 1-3 months lag
            full_df[f"{var}_lag{lag}"] = full_df.groupby("location")[var].shift(lag)

    # Log-transform incidence (avoid skew)
    full_df["malaria_incidence_log"] = np.log1p(full_df["malaria_incidence"])

    df_newer = full_df[full_df["date"] >= earliest_time] # only keeps the future timepoints, but now with lagged variables from historic data

    # Define features
    features = [
        "rainfall","mean_temperature",
        "rainfall_lag1", "rainfall_lag2", "rainfall_lag3",
        "mean_temperature_lag1","mean_temperature_lag2","mean_temperature_lag3",
        "malaria_incidence_lag1","malaria_incidence_lag2","malaria_incidence_lag3"
    ]

    X_new = df_newer[features]

    rf_model = joblib.load(model_fn)

    y_pred_log = rf_model.predict(X_new) #this is the predicted malaria incidence on the log scale

    y_pred = np.expm1(y_pred_log)   # this is the predicted malaria incidence per 10000

    df_newer["malaria_incidence"] = y_pred

    df_newer["sample_0"] = df_newer["malaria_incidence"]*df_newer["population"]/10000 #for samples in casenumbers for each district

    #Save predictions to file
    df_newer.to_csv(predictions_fn, index=False)

    #the predictions are now further ahead than the lags, also no rolling window prediction yet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)
