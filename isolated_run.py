from predict import predict
from train import train

train("input/training_data.csv", "output/model.bin")
predict("output/model.bin", "input/training_data.csv", "input/future_data.csv", "output/predictions.csv")