import argparse
import joblib
import pandas as pd

# this is currently just copied from the rwanda_SARIMAX model, needs to be altered

def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    # get all unique districts from historic data
    df = pd.read_csv(future_climatedata_fn)
    districts = pd.read_csv(historic_data_fn)['location'].unique()

    final_predictions_df = pd.DataFrame()

    for district in districts:
        model_file_name = model_fn + "_" + district + ".bin"
        model = joblib.load(model_file_name)
        future_data_for_district = df[df['location'] == district]
        print(f"Loaded model for district {district} from {model_file_name}")
        predictions = model.forecast(steps=len(future_data_for_district), exog=future_data_for_district[['rainfall', 'mean_temperature']])
        samples_0 = predictions
        #predictions = pd.Series(predictions, index=test_data.index)

        # put future data for district into final_predictions_df with a new column 'sample_0' containing the predictions
        future_data_for_district['sample_0'] = samples_0
        final_predictions_df = pd.concat([final_predictions_df, future_data_for_district])

    final_predictions_df.to_csv(predictions_fn, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)
