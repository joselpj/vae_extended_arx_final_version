import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from random import randint
import os
import datetime


def load_dataset(dataset_filename):
    df = pd.read_csv(os.path.join("data", dataset_filename), sep=",")
    dataset = df.iloc[:, :].values
    return dataset


def cascaded_tanks_dataset(na=1, nb=1, nk=1, normalize=True, data_train=True, window_dim=50):
    dataset = load_dataset("cascaded_tanks.csv")  # original dataset

    samples = 1000
    rows = int((samples / window_dim) * (window_dim - max(na, nb)))

    uk_train = dataset[0:samples, 0:1]  # uEst in original csv file
    yk_train = dataset[0:samples, 2:3]  # yEst in original csv file

    uk_test = dataset[0:samples, 1:2]  # uVal in original csv file
    yk_test = dataset[0:samples, 3:4]  # yVal in original csv file

    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        uk_train = scaler.fit_transform(uk_train)
        yk_train = scaler.fit_transform(yk_train)
        uk_test = scaler.fit_transform(uk_test)
        yk_test = scaler.fit_transform(yk_test)

    train_inputs = np.zeros((rows, (window_dim * 2) + na + nb))
    train_outputs = np.zeros((rows, 1))

    test_inputs = np.zeros((rows, (window_dim * 2) + na + nb))
    test_outputs = np.zeros((rows, 1))

    windows_est = np.zeros((int(samples / window_dim), window_dim * 2))
    windows_val = np.zeros((int(samples / window_dim), window_dim * 2))

    for i in range(int(samples / window_dim)):
        init_index = i * (window_dim)  # init_index = i * (window_dim-1)
        end_index = init_index + window_dim
        yk_est = yk_train[init_index:end_index, :]
        yk_est = yk_est.reshape((1, yk_est.shape[0]))

        yk_val = yk_test[init_index:end_index, :]
        yk_val = yk_val.reshape((1, yk_val.shape[0]))

        uk_est = uk_train[init_index:end_index, :]
        uk_est = uk_est.reshape((1, uk_est.shape[0]))

        uk_val = uk_test[init_index:end_index, :]
        uk_val = uk_val.reshape((1, uk_val.shape[0]))

        full_input_est = np.concatenate((yk_est, uk_est), axis=1)
        full_input_val = np.concatenate((yk_val, uk_val), axis=1)

        windows_est[i, :] = full_input_est
        windows_val[i, :] = full_input_val

    for i in range(int(samples / window_dim)):
        train_inputs[i * (window_dim - max(na, nb)):(i + 1) * (window_dim - max(na, nb)), 0:(window_dim * 2)] = windows_est[i, :]
        test_inputs[i * (window_dim - max(na, nb)):(i + 1) * (window_dim - max(na, nb)), 0:(window_dim * 2)] = windows_val[i, :]

    #   ------------------------------------------
    #   to fill y(k-1), y(k-2), y(k-3), y(k-4)
    #   ------------------------------------------
    init_sample_y = na - 1
    init_sample_u = nb - 1
    for i in range(rows):
        if i % (window_dim-max(na, nb)) == 0:
            init_sample_y = na - 1
        init_sample_y = init_sample_y + 1

        for j in range(na):
            train_inputs[i, window_dim * 2 + j] = train_inputs[i, init_sample_y - j]
            test_inputs[i, window_dim * 2 + j] = test_inputs[i, init_sample_y - j]

        #   ------------------------------------------
        #   to fill u(k-1), u(k-2), u(k-3), u(k-4),... u(k-nb)
        #   ------------------------------------------
        if i % (window_dim-max(na, nb)) == 0:
            init_sample_u = na - 1
        init_sample_u = init_sample_u + 1
        for j in range(nb):
            train_inputs[i, window_dim * 2 + j + na] = train_inputs[i, window_dim + (init_sample_u - j)]
            test_inputs[i, window_dim * 2 + j + na] = test_inputs[i, window_dim + (init_sample_u - j)]

        train_outputs[i, :] = train_inputs[i, init_sample_y + 1]
        test_outputs[i, :] = test_inputs[i, init_sample_y + 1]

    return train_inputs, train_outputs, test_inputs, test_outputs


def gas_furnace_dataset(na=1, nb=1, nk=1, normalize=True, data_train=True, window_dim=50):
    dataset = load_dataset("gas-furnace.csv")

    samples = 200
    rows = int((samples / window_dim) * (window_dim - max(na, nb)))

    uk_train = dataset[0:samples, 0:1]  # InputGasRate in original csv file
    yk_train = dataset[0:samples, 1:2]  # CO2 in original csv file

    uk_test = dataset[samples:, 0:1]  # InputGasRate in original csv file
    yk_test = dataset[samples:, 1:2]  # CO2 in original csv file

    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        uk_train = scaler.fit_transform(uk_train)
        yk_train = scaler.fit_transform(yk_train)
        uk_test = scaler.fit_transform(uk_test)
        yk_test = scaler.fit_transform(yk_test)

    train_inputs = np.zeros((rows, (window_dim * 2) + na + nb))
    train_outputs = np.zeros((rows, 1))

    test_inputs = np.zeros((uk_test.shape[0], (window_dim * 2) + na + nb))
    test_outputs = np.zeros((yk_test.shape[0], 1))

    windows_est = np.zeros((int(samples / window_dim), window_dim * 2))
    windows_val = np.zeros((1, window_dim * 2))

    # the following code allows constructing train inputs and outputs
    # for to select input windows
    for i in range(int(samples / window_dim)):
        init_index = i * window_dim  # init_index = i * (window_dim-1)
        end_index = init_index + window_dim
        yk_est = yk_train[init_index:end_index, :]
        yk_est = yk_est.reshape((1, yk_est.shape[0]))

        uk_est = uk_train[init_index:end_index, :]
        uk_est = uk_est.reshape((1, uk_est.shape[0]))

        full_input_est = np.concatenate((yk_est, uk_est), axis=1)

        windows_est[i, :] = full_input_est

    # for to fill train_inputs[row,0:50] with windows
    for i in range(int(samples / window_dim)):
        train_inputs[i * (window_dim - max(na, nb)):(i + 1) * (window_dim - max(na, nb)), 0:(window_dim * 2)] = windows_est[i, :]

    #   ------------------------------------------
    #   to fill y(k-1), y(k-2), y(k-3), y(k-4)
    #   ------------------------------------------
    init_sample_y = na - 1
    init_sample_u = nb - 1
    for i in range(rows):
        if i % (window_dim - max(na, nb)) == 0:
            init_sample_y = na - 1
        init_sample_y = init_sample_y + 1

        for j in range(na):
            train_inputs[i, window_dim * 2 + j] = train_inputs[i, init_sample_y - j]

        #   ------------------------------------------
        #   to fill u(k-1), u(k-2), u(k-3), u(k-4),... u(k-nb)
        #   ------------------------------------------
        if i % (window_dim - max(na, nb)) == 0:
            init_sample_u = na - 1
        init_sample_u = init_sample_u + 1
        for j in range(nb):
            train_inputs[i, window_dim * 2 + j + na] = train_inputs[i, window_dim + (init_sample_u - j)]

        train_outputs[i, :] = train_inputs[i, init_sample_y + 1]

    # the following code allows constructing test inputs and outputs
    # for to select input windows
    for i in range(1):
        init_index = i * window_dim  # init_index = i * (window_dim-1)
        end_index = init_index + window_dim
        yk_val = yk_test[init_index:end_index, :]
        yk_val = yk_val.reshape((1, yk_val.shape[0]))

        uk_val = uk_test[init_index:end_index, :]
        uk_val = uk_val.reshape((1, uk_val.shape[0]))

        full_input_val = np.concatenate((yk_val, uk_val), axis=1)

        windows_val[i, :] = full_input_val

    # for to fill train_inputs[row,0:50] with windows
    for i in range(1):
        test_inputs[i * (window_dim - max(na, nb)):(i + 1) * (window_dim - max(na, nb)), 0:(window_dim * 2)] = windows_val[i, :]

        #   ------------------------------------------
        #   to fill y(k-1), y(k-2), y(k-3), y(k-4)
        #   ------------------------------------------
        init_sample_y = na - 1
        init_sample_u = nb - 1
        for i in range(test_inputs.shape[0]):
            if i % (window_dim - max(na, nb)) == 0:
                init_sample_y = na - 1
            init_sample_y = init_sample_y + 1

            for j in range(na):

                test_inputs[i, window_dim * 2 + j] = test_inputs[i, init_sample_y - j]

            #   ------------------------------------------
            #   to fill u(k-1), u(k-2), u(k-3), u(k-4),... u(k-nb)
            #   ------------------------------------------
            if i % (window_dim - max(na, nb)) == 0:
                init_sample_u = na - 1
            init_sample_u = init_sample_u + 1
            for j in range(nb):

                test_inputs[i, window_dim * 2 + j + na] = test_inputs[i, window_dim + (init_sample_u - j)]


            test_outputs[i, :] = test_inputs[i, init_sample_y + 1]

    return train_inputs, train_outputs, test_inputs, test_outputs


def silverbox_dataset(na=1, nb=1, nk=1, normalize=True, data_train=True, window_dim=50):
    dataset = load_dataset("Schroeder80mV.csv")  # original dataset

    samples = 1000
    rows = int((samples / window_dim) * (window_dim - max(na, nb)))

    uk_train = dataset[0:samples, 2:3]  # V1 in original csv file
    yk_train = dataset[0:samples, 3:4]  # V2 in original csv file

    uk_test = dataset[samples:samples * 2, 2:3]  # V1 in original csv file
    yk_test = dataset[samples:samples * 2, 3:4]  # V2 in original csv file

    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        uk_train = scaler.fit_transform(uk_train)
        yk_train = scaler.fit_transform(yk_train)
        uk_test = scaler.fit_transform(uk_test)
        yk_test = scaler.fit_transform(yk_test)

    train_inputs = np.zeros((rows, (window_dim * 2) + na + nb))
    train_outputs = np.zeros((rows, 1))

    test_inputs = np.zeros((rows, (window_dim * 2) + na + nb))
    test_outputs = np.zeros((rows, 1))

    windows_est = np.zeros((int(samples / window_dim), window_dim * 2))
    windows_val = np.zeros((int(samples / window_dim), window_dim * 2))

    for i in range(int(samples / window_dim)):
        init_index = i * window_dim  # init_index = i * (window_dim-1)
        end_index = init_index + window_dim
        yk_est = yk_train[init_index:end_index, :]
        yk_est = yk_est.reshape((1, yk_est.shape[0]))

        yk_val = yk_test[init_index:end_index, :]
        yk_val = yk_val.reshape((1, yk_val.shape[0]))

        uk_est = uk_train[init_index:end_index, :]
        uk_est = uk_est.reshape((1, uk_est.shape[0]))

        uk_val = uk_test[init_index:end_index, :]
        uk_val = uk_val.reshape((1, uk_val.shape[0]))

        full_input_est = np.concatenate((yk_est, uk_est), axis=1)
        full_input_val = np.concatenate((yk_val, uk_val), axis=1)

        windows_est[i, :] = full_input_est
        windows_val[i, :] = full_input_val

    for i in range(int(samples / window_dim)):
        train_inputs[i * (window_dim - max(na, nb)):(i + 1) * (window_dim - max(na, nb)), 0:(window_dim * 2)] = windows_est[i, :]
        test_inputs[i * (window_dim - max(na, nb)):(i + 1) * (window_dim - max(na, nb)), 0:(window_dim * 2)] = windows_val[i, :]

    #   ------------------------------------------
    #   to fill y(k-1), y(k-2), y(k-3), y(k-4)
    #   ------------------------------------------
    init_sample_y = na - 1
    init_sample_u = nb - 1
    for i in range(rows):
        if i % (window_dim-max(na, nb)) == 0:
            init_sample_y = na - 1
        init_sample_y = init_sample_y + 1

        for j in range(na):
            train_inputs[i, window_dim * 2 + j] = train_inputs[i, init_sample_y - j]
            test_inputs[i, window_dim * 2 + j] = test_inputs[i, init_sample_y - j]

        #   ------------------------------------------
        #   to fill u(k-1), u(k-2), u(k-3), u(k-4),... u(k-nb)
        #   ------------------------------------------
        if i % (window_dim-max(na, nb)) == 0:
            init_sample_u = na - 1
        init_sample_u = init_sample_u + 1
        for j in range(nb):
            train_inputs[i, window_dim * 2 + j + na] = train_inputs[i, window_dim + (init_sample_u - j)]
            test_inputs[i, window_dim * 2 + j + na] = test_inputs[i, window_dim + (init_sample_u - j)]

        train_outputs[i, :] = train_inputs[i, init_sample_y + 1]
        test_outputs[i, :] = test_inputs[i, init_sample_y + 1]

    return train_inputs, train_outputs, test_inputs, test_outputs


def wiener_hammer_dataset(na=1, nb=1, nk=1, normalize=True, data_train=True, window_dim=50):
    dataset = load_dataset("WienerHammerBenchmark.csv")  # original dataset

    samples = 1000
    rows = int((samples / window_dim) * (window_dim - max(na, nb)))

    uk_train = dataset[samples*5:samples*6, 0:1]  # uBenchMark in original csv file
    yk_train = dataset[samples*5:samples*6, 1:2]  # yBenchMark in original csv file

    uk_test = dataset[samples*6:samples*7, 0:1]  # uBenchMark in original csv file
    yk_test = dataset[samples*6:samples*7, 1:2]  # yBenchMark in original csv file

    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        uk_train = scaler.fit_transform(uk_train)
        yk_train = scaler.fit_transform(yk_train)
        uk_test = scaler.fit_transform(uk_test)
        yk_test = scaler.fit_transform(yk_test)

    train_inputs = np.zeros((rows, (window_dim * 2) + na + nb))
    train_outputs = np.zeros((rows, 1))

    test_inputs = np.zeros((rows, (window_dim * 2) + na + nb))
    test_outputs = np.zeros((rows, 1))

    windows_est = np.zeros((int(samples / window_dim), window_dim * 2))
    windows_val = np.zeros((int(samples / window_dim), window_dim * 2))

    for i in range(int(samples / window_dim)):
        init_index = i * window_dim  # init_index = i * (window_dim-1)
        end_index = init_index + window_dim
        yk_est = yk_train[init_index:end_index, :]
        yk_est = yk_est.reshape((1, yk_est.shape[0]))

        yk_val = yk_test[init_index:end_index, :]
        yk_val = yk_val.reshape((1, yk_val.shape[0]))

        uk_est = uk_train[init_index:end_index, :]
        uk_est = uk_est.reshape((1, uk_est.shape[0]))

        uk_val = uk_test[init_index:end_index, :]
        uk_val = uk_val.reshape((1, uk_val.shape[0]))

        full_input_est = np.concatenate((yk_est, uk_est), axis=1)
        full_input_val = np.concatenate((yk_val, uk_val), axis=1)

        windows_est[i, :] = full_input_est
        windows_val[i, :] = full_input_val

    for i in range(int(samples / window_dim)):
        train_inputs[i * (window_dim - max(na, nb)):(i + 1) * (window_dim - max(na, nb)), 0:(window_dim * 2)] = windows_est[i, :]
        test_inputs[i * (window_dim - max(na, nb)):(i + 1) * (window_dim - max(na, nb)), 0:(window_dim * 2)] = windows_val[i, :]

    #   ------------------------------------------
    #   to fill y(k-1), y(k-2), y(k-3), y(k-4)
    #   ------------------------------------------
    init_sample_y = na - 1
    init_sample_u = nb - 1
    for i in range(rows):
        if i % (window_dim-max(na, nb)) == 0:
            init_sample_y = na - 1
        init_sample_y = init_sample_y + 1

        for j in range(na):
            train_inputs[i, window_dim * 2 + j] = train_inputs[i, init_sample_y - j]
            test_inputs[i, window_dim * 2 + j] = test_inputs[i, init_sample_y - j]

        #   ------------------------------------------
        #   to fill u(k-1), u(k-2), u(k-3), u(k-4),... u(k-nb)
        #   ------------------------------------------
        if i % (window_dim-max(na, nb)) == 0:
            init_sample_u = na - 1
        init_sample_u = init_sample_u + 1
        for j in range(nb):
            train_inputs[i, window_dim * 2 + j + na] = train_inputs[i, window_dim + (init_sample_u - j)]
            test_inputs[i, window_dim * 2 + j + na] = test_inputs[i, window_dim + (init_sample_u - j)]

        train_outputs[i, :] = train_inputs[i, init_sample_y + 1]
        test_outputs[i, :] = test_inputs[i, init_sample_y + 1]

    return train_inputs, train_outputs, test_inputs, test_outputs

if __name__ == "__main__":
    train_inputs, train_outputs, test_inputs, test_outputs = wiener_hammer_dataset(4, 5, normalize=False)
    print("row 0:\n ",train_inputs[0])
    print("row 44:\n ",train_inputs[44])
    #print(test_inputs[0])
    #print(test_inputs[44])
    #print(train_outputs[0])
    #print(train_outputs[44])
    #print(test_outputs[0])
    #print(test_outputs[44])
    #np.savetxt("data/" + "train_inputs.csv", train_inputs, delimiter=',')
    #np.savetxt("data/" + "train_outputs.csv", train_outputs, delimiter=',')
    #np.savetxt("data/" + "test_inputs.csv", test_inputs, delimiter=',')
    #np.savetxt("data/" + "test_outputs.csv", test_outputs, delimiter=',')
