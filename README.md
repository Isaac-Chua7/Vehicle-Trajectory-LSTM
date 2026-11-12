# Vehicle-Trajectory-LSTM
This is my vehicle trajectory LSTM repository, which, after observing 2 seconds of a vehicle's motion, predicts the trajectory of the vehicle  over a 3-second horizon, as proposed in my paper "Trajectory Prediction of Road Vehicles for Proactive Cyclist Crash Avoidance and Risk Assessment"

#Instructions For Use
Ensure LSTM_Practice.ipnb, rbd_data.csv, and the 4-feature-models folder are downloaded; these files and this folder should be in the same root folder.

When running the notebook, ensure to run cells in order from top to bottom as some cells are dependent on those above. The current configuration loads the pretrained final model: (4 input features, 128 hidden layer size, 2 hidden layers and 100 epochs).

However, modification or omission of the line below can be used to load a different model or train a new one:
(Under heading "Instantiating Model")
"print(model.load_state_dict(torch.load('4-feature-models/h128_n2_e100_f4.pt', weights_only=True)))"

When training a new model, ensure run_epochs = True under the heading "Multiple Epochs".
