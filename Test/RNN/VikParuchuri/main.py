from sklearn.preprocessing import StandardScaler
import math
import functions as fn
import pandas as pd
import numpy as np

breakpoint()

os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir)) #go back to Test folder
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir)) ##go back to main folder

# Read in our data, and fill missing values by repeting last entry
data = pd.read_csv("Data_Input/clean_weather.csv", index_col=0)
data = data.ffill()

# Define predictors and target
PREDICTORS = ["tmax", "tmin", "rain"]
TARGET = "tmax_tomorrow"

# Scale our data to have mean 0
scaler = StandardScaler()
data[PREDICTORS] = scaler.fit_transform(data[PREDICTORS])

# Split into train, valid, test sets
np.random.seed(0)
split_data = np.split(data, [int(.7*len(data)), int(.85*len(data))])
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in split_data]





breakpoint()
        ####    TRAINING LOOP   ####

#an epoch in the training loop refers to one complete pass through the entire training dataset
epochs = 250
lr = 1e-5

# we take out four features, generate 4 input features. And output a single feature.
layer_conf = [
    {"type":"input", "units": 3},
    {"type": "rnn", "hidden": 4, "output": 1}
]
layers = fn.init_params(layer_conf)

for epoch in range(epochs):
    sequence_len = 7 #how long of a sequence we want to feed into our rnn
    epoch_loss = 0 #track the loss epoch by epoch
    for j in range(train_x.shape[0] - sequence_len):
        seq_x = train_x[j:(j+sequence_len),]
        seq_y = train_y[j:(j+sequence_len),]
        hiddens, outputs = fn.forward(seq_x, layers)
        grad = fn.mse_grad(seq_y, outputs)
        params = fn.backward(layers, seq_x, lr, grad, hiddens)
        epoch_loss += fn.mse(seq_y, outputs)

    if epoch % 50 == 0:
        sequence_len = 7
        valid_loss = 0
        for j in range(valid_x.shape[0] - sequence_len):
            seq_x = valid_x[j:(j+sequence_len),]
            seq_y = valid_y[j:(j+sequence_len),]
            _, outputs = fn.forward(seq_x, layers)
            valid_loss += fn.mse(seq_y, outputs)

        print(f"Epoch: {epoch} train loss {epoch_loss / len(train_x)} valid loss {valid_loss / len(valid_x)}")
