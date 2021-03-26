import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data_utils
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from validation_test import *
from tensorboardX import SummaryWriter

def prepare_data(X_train_final, y_train, X_test_final, y_test, batch_size=32, num_workers=1):
    """
    Converts numpy objects to torch objects, using Torch DataLoader

    params: batch_size - default 32
    params: num_workers - default 1, speeds up training but slows machine
    params: X_train_final - dataset after pipeline transform
    params: y_train - labels
    params: X_test_final - dataset after pipeline transform
    params: y_test - labels

    returns:
    X_train_final_torch - [torch dataset] - includes labels
    y_train_torch - [torch dataset] - labels
    X_test_final_torch - [torch dataset] - includes labels
    y_test_torch - [torch dataset] - labels
    """
    X_train_final_torch  = torch.tensor(X_train_final)
    y_train_torch = torch.IntTensor(y_train.to_numpy())

    X_train_torch =TensorDataset(X_train_final_torch, y_train_torch)
    X_train_final_torch = DataLoader(X_train_torch, batch_size, shuffle = False)


    X_test_final_torch  = torch.tensor(X_test_final)
    y_test_torch = torch.IntTensor(y_test.to_numpy())

    X_test_torch =TensorDataset(X_test_final_torch, y_test_torch)
    X_test_final_torch = DataLoader(X_test_torch, batch_size, shuffle = False)

    return X_train_final_torch, y_train_torch, X_test_final_torch, y_test_torch

class Torch_model(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, solver, activation):
        super(Torch_model, self).__init__()
        layers = []
        for _ in range(n_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(X_train_final_torch, X_test_final_torch, model, epochs = 100, learning_rate=0.01, patience = 5):
    """
    params: epochs [int] - default 100
    params: X_train_final_torch [torch dataset] - data and labels
    params: model [torch object] - torch ML model
    params: learning_rate [int] - default 0.001, Adam optim

    returns: model [torch object] - TRAINED torch ML model
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    validation = Validation()

    writer = SummaryWriter()

    print("Starting training....")
    for epoch in range(epochs):
        epoch_error = 0
        val_error = 0
        break_training = False
        # train data
        for i, batch in enumerate(X_train_final_torch):
            inputs, labels = batch
            optimizer.zero_grad()
            out = model(inputs.float())
            loss = criterion(out, labels.float().view(-1,1))
            loss.backward()
            optimizer.step()
            epoch_error += round(np.sqrt(np.absolute(loss.item())), 0)
        
        #validate data
        for i, batch in enumerate(X_test_final_torch):
            inputs, labels = batch
            prediction = model(inputs.float())
            val_loss = criterion(prediction, labels.float().view(-1, 1))
            val_error += round(np.sqrt(np.absolute(val_loss.item())), 0)

        model_best, best_val_error, break_training = validation.validate_testing(model, val_error, patience=patience)
        
        if break_training == True:
            model = model_best
            print(f"val error hasent improved over time, breaking the training at val {best_val_error}")
            return model

        writer.add_scalar("epoch error", epoch_error, epoch)
        writer.add_scalar("val error", val_error, epoch)
    
        print("epoch",(epoch + 1),"/",epochs, "train loss: ", epoch_error, "val loss: ", val_error)
    print("Training finished")
    writer.close()
    return model

