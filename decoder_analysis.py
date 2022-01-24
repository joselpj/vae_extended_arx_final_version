import torch
import numpy as np
from model import VaeArx
from model_decoder import Decoder
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from load_data import cascaded_tanks_dataset
# from load_data import gas_furnace_dataset
# from load_data import silverbox_dataset
# from load_data import wiener_hammer_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Load trained model
vaearx = VaeArx(100, 1, 2, 64, 2)
vaearx.load_state_dict(torch.load("trained_models/vaearx_cascaded_tanks_dataset2021-10-12_13-10-18.dat"))

print(vaearx)

# -----------------------------------------------
# decoder
# -----------------------------------------------
decoder = Decoder(2, 1, 64, 2)
dec1_layer = vaearx.dec1.weight
dec2_layer = vaearx.dec2.weight
out_layer = vaearx.out.weight

dec1_bias = vaearx.dec1.bias
dec2_bias = vaearx.dec2.bias
out_bias = vaearx.out.bias

decoder.dec1.weight = torch.nn.Parameter(dec1_layer)
decoder.dec2.weight = torch.nn.Parameter(dec2_layer)
decoder.out.weight = torch.nn.Parameter(out_layer)

decoder.dec1.bias = torch.nn.Parameter(dec1_bias)
decoder.dec2.bias = torch.nn.Parameter(dec2_bias)
decoder.out.bias = torch.nn.Parameter(out_bias)

# --------------------------------------------------------------------------------
# load test data
# --------------------------------------------------------------------------------
data_name = "cascaded_tanks_dataset"
x_train, y_train, x_test, y_test = cascaded_tanks_dataset(4, 5, 1, normalize=False)
test_inputs = torch.Tensor(x_test)
test_outputs = torch.Tensor(y_test)

output = []
samples = test_outputs.shape[0]
columns = 2  # test_inputs.shape[1]

code = [[-0.1, -0.2]]
z_code = torch.tensor(code)
z_code = z_code.reshape((1, 2))
for i in range(samples):
    x_in = test_inputs[i, 100:102]
    x_in = torch.Tensor(x_in).reshape(1, columns)
    dec_input = torch.cat((x_in, z_code), 1)
    output.append(decoder.forward(dec_input).detach().numpy()[0, 0])

output_arr = np.array(output)

"""""
# revert normalization. data in original range of values.
max = 10  # cascaded_tank: 10 ; gas_furnace: 60.5 ; silverbox :0.26493 ; wiener_hammer: 0.63587
min = 2.9116  # cascaded_tank: 2.9116 ; gas_furnace: 45.6 ; silverbox :-0.26249 ; wiener_hammer: -1.1203
scaler = MinMaxScaler(feature_range=(min, max))
output_arr = scaler.fit_transform(output_arr.reshape(samples, 1))
test_outputs = scaler.fit_transform(test_outputs.reshape(samples, 1))
"""""


MSE = mean_squared_error(output_arr, test_outputs)
print(MSE)


def plot_vaearx_reconstruction(y_name, legend_loc):
    plt.plot(test_outputs, 'b-', label='validation data', linewidth=1)
    plt.plot(output_arr, 'r--', label='decoder reconstruction', linewidth=1)
    plt.title('Output')
    plt.xlabel('samples')
    plt.ylabel(y_name)
    plt.grid()
    plt.legend(loc=legend_loc)
    plt.show()


plot_vaearx_reconstruction(y_name='magnitude (V)', legend_loc='lower center')
