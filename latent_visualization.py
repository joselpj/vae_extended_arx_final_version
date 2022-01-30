import torch
import numpy as np
from model import VaeArx
from model_decoder import Decoder
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# from load_data import cascaded_tanks_dataset
# from load_data import gas_furnace_dataset
# from load_data import silverbox_dataset
from load_data import wiener_hammer_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Load trained model
vaearx = VaeArx(100, 1, 2, 64, 9)
vaearx.load_state_dict(torch.load("trained_models/vaearx_wiener_hammer_dataset2021-11-10_21-33-41.dat"))

print(vaearx)

data_name = "wiener_hammer_dataset"
x_train, y_train, x_test, y_test = wiener_hammer_dataset(4, 5, 1, normalize=False)
test_inputs = torch.Tensor(x_test)
test_outputs = torch.Tensor(y_test)

output = []
latent = []
samples = test_outputs.shape[0]
columns = test_inputs.shape[1]

for i in range(samples):
    x_in = test_inputs[i, :]
    x_in = torch.Tensor(x_in).reshape(1, columns)
    output.append(vaearx.forward(x_in).detach().numpy()[0, 0])
    latent.append(vaearx.latent_r.to('cpu').detach().numpy())

output_arr = np.array(output)
latent_arr = np.array(latent)
print(latent_arr.shape)
latent_arr = latent_arr.reshape((900,2))
print(latent_arr.shape)



"""""
# revert normalization. data in original range of values.
max = 10 # cascaded_tank: 10 ; gas_furnace: 60.5 ; silverbox :0.26493 ; wiener_hammer: 0.63587
min = 2.9116 # cascaded_tank: 2.9116 ; gas_furnace: 45.6 ; silverbox :-0.26249 ; wiener_hammer: -1.1203
scaler = MinMaxScaler(feature_range=(min, max))
output_arr = scaler.fit_transform(output_arr.reshape(samples, 1))
test_outputs = scaler.fit_transform(test_outputs.reshape(samples, 1))
"""""




MSE = mean_squared_error(output_arr, test_outputs)
print(MSE)


def plot_vaearx_reconstruction(y_name, legend_loc):
    plt.plot(test_outputs[0:200], 'b-', label='validation data', linewidth=1)
    plt.plot(output_arr[0:200], 'r--', label='vaearx reconstruction', linewidth=1)
    plt.title('Output')
    plt.xlabel('samples')
    plt.ylabel(y_name)
    plt.grid()
    plt.legend(loc=legend_loc)
    plt.show()


plot_vaearx_reconstruction(y_name='V2', legend_loc='lower right')

from scipy.stats import norm

x = np.linspace(-3, 3, 300)

fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(hspace=0.6, wspace=0.4)

for i in range(2):
    ax = fig.add_subplot(1, 2, i+1)
    ax.hist(latent_arr[:,i], density=True, bins = 20, label="$\mu$:" + str(np.mean(latent_arr[:,i])) + "\n$\sigma:$" + str(np.std(latent_arr[:, i])))
    print("\nMean: ", np.mean(latent_arr[:,i]))
    print("\nstd: ", np.std(latent_arr[:, i]))
    #ax.axis('off')
    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
    ax.plot(x,norm.pdf(x), label='std_norm_pdf')
    ax.legend(prop={"size":16})
    plt.grid(True)


plt.show()