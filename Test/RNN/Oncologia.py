import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from VikParuchuri import functions as fn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import train_test_split
# np.random.seed(0)


os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir)) #go back to Test folder
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir)) ##go back to main folder


# Read in our data, and fill missing values by repeting last entry
data = pd.read_csv('Input/Oncologia/Ratios/RATIOrnn.csv')
n_ciclos = 4
# Define predictors and target
PREDICTORS = ["Linfocitos", "Neutrofilos", "Plaquetas"]
TARGET = "Vivo"

# Scale our data to have mean 0
scaler = StandardScaler()
data[PREDICTORS] = scaler.fit_transform(data[PREDICTORS])

n_pacients = int(len(data)/n_ciclos)

def generate_unique_random_numbers(count, min_value, max_value):
    if max_value - min_value + 1 < count:
        raise ValueError("Not enough unique numbers in the specified range")
    return random.sample(range(min_value, max_value + 1), count)


# [test_pacient,valid_pacient] = generate_unique_random_numbers(2, 1, n_pacients)
test_pacient = random.randint(1, n_pacients)
valid_pacient = random.randint(1, n_pacients-1)

test_pacient = 1
valid_pacient = n_pacients-1


X = data.loc[:, data.columns != TARGET]
y = data[TARGET]

X = np.roll(X, (n_pacients-test_pacient)*n_ciclos, axis=0)
y = np.roll(y, (n_pacients-test_pacient)*n_ciclos, axis=0)

X2, test_x, y2, test_y = train_test_split(X, y, test_size=n_ciclos, shuffle = False)

X2 = np.roll(X2, (n_pacients-valid_pacient)*n_ciclos, axis=0)
y2 = np.roll(y2, (n_pacients-valid_pacient)*n_ciclos, axis=0)

train_x, valid_x, train_y, valid_y = train_test_split(X2, y2, test_size=n_ciclos, shuffle = False)

train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)
valid_y = valid_y.reshape(-1, 1)


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
    # breakpoint()
    sequence_len = n_ciclos #how long of a sequence we want to feed into our rnn
    epoch_loss = 0 #track the loss epoch by epoch
    for j in range(train_x.shape[0] - sequence_len):
        seq_x = train_x[j:(j+sequence_len),]
        seq_y = train_y[j:(j+sequence_len),]
        hiddens, outputs = fn.forward(seq_x, layers)
        grad = fn.mse_grad(seq_y, outputs)
        # breakpoint()
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

breakpoint()