import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import VaeArx
from utils import target_loss
import os
import datetime

#from load_data import cascaded_tanks_dataset
#from load_data import gas_furnace_dataset
#from load_data import silverbox_dataset
from load_data import wiener_hammer_dataset


device = torch.device("cuda:0")

# Setup model with 2 latent neurons
vaearx = VaeArx(100, 1, 2, 64, 9)

# Load and prepare training data
data_name = "wiener_hammer_dataset"
x_train, y_train, x_test, y_test = wiener_hammer_dataset(4, 5, 1, normalize=False)
inputs = torch.FloatTensor(x_train)
outputs = torch.FloatTensor(y_train)

traindata = TensorDataset(inputs, outputs)
dataloader = DataLoader(traindata, batch_size=1, shuffle=False, num_workers=1)

# Training setup
SAVE_PATH = "trained_models/vaearx_"
N_EPOCHS = 100
optimizer = optim.Adam(vaearx.parameters())
hist_error = []
hist_loss = []
beta = 0.5

# Training loop
for epoch in range(N_EPOCHS):  
    epoch_error = []
    epoch_loss = []
    for i_batch, minibatch in enumerate(dataloader):

        inputs, outputs = minibatch
        optimizer.zero_grad()
        pred = vaearx.forward(inputs)
        
        loss = target_loss(pred, outputs) + beta * vaearx.kl_loss
        loss.backward()
        optimizer.step()
        error = torch.mean(torch.sqrt((pred[:,0]-outputs)**2)).detach().numpy()
        epoch_error.append(error)
        epoch_loss.append(loss.data.detach().numpy())
    hist_error.append(np.mean(epoch_error))
    hist_loss.append(np.mean(epoch_loss))
    print("Epoch %d -- loss %f, RMS error %f " % (epoch+1, hist_loss[-1], hist_error[-1]))
    
torch.save(vaearx.state_dict(), SAVE_PATH + data_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".dat")
print("Model saved to %s" % SAVE_PATH + data_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".dat")

plt.plot(hist_error)
plt.title("Training RMS Error")
plt.xlabel('Epochs')
plt.grid()
plt.show()

plt.plot(hist_loss)
plt.title("Training MSE Loss")
plt.xlabel('Epochs')
plt.grid()
plt.show()
