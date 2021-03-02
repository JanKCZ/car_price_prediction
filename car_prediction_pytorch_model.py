import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data_utils
import numpy as np

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
    X_train_final_torch = DataLoader(X_train_torch, batch_size, shuffle = True)


    X_test_final_torch  = torch.tensor(X_test_final)
    y_test_torch = torch.IntTensor(y_test.to_numpy())

    X_test_torch =TensorDataset(X_test_final_torch, y_test_torch)
    X_test_final_torch = DataLoader(X_test_torch, batch_size, shuffle = True)

    return X_train_final_torch, y_train_torch, X_test_final_torch, y_test_torch

class nnModel(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(nnModel, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.relu3 = nn.ReLU()
		self.fc4 = nn.Linear(hidden_size, 1)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu1(out)
		out = self.fc2(out)
		out = self.relu2(out)
		out = self.fc3(out)
		out = self.relu3(out)
		out = self.fc4(out)
		return out


# model = nnModel(X_train_final.shape[1], 800)


def train_model(X_train_final_torch, model, epochs = 100, learning_rate=0.001):
    """
    params: epochs [int] - default 100
    params: X_train_final_torch [torch dataset] - data and labels
    params: model [torch object] - torch ML model
    params: learning_rate [int] - default 0.001, Adam optim

    returns: model [torch object] - TRAINED torch ML model
    """
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    print("Starting training....")
    for epoch in range(epochs):
        epoch_error = 0
        for i, batch in enumerate(X_train_final_torch):
            inputs, labels = batch
            optimizer.zero_grad()
            out = model(inputs.float())
            loss = criterion(out, labels.float().view(-1,1))
            loss.backward()
            optimizer.step()
            epoch_error += round(np.sqrt(np.absolute(loss.item())), 0)
        print("train MSE loss: ", epoch_error)
    print("Training finished")