import os

import json
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dense, RNN
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from larnn import LinearAntisymmetricCell


def rolling_window(x, y, steps, step_size=None):
    nsamp = x.shape[0]
    nfeatures = x.shape[1]
    rollx = np.zeros((nsamp - steps + 1, steps, nfeatures))
    if step_size is not None:
        rollstep= np.zeros((nsamp - steps + 1, steps))

    for i in range(steps):
        rollx[:, i, :] = x[i:nsamp - steps + i + 1, :]
        if step_size is not None:
            rollstep[:, i] = step_size[i:nsamp - steps + i + 1]
    rolly = y[steps - 1:]
    if step_size is not None:
        return rollx, rolly, rollstep
    else:
        return rollx, rolly


def split_data(x, y, batch_size, train_test_ratio=0.8, step_sizes=None):
    input_len = x.shape[0]
    to_train = int(input_len * train_test_ratio)
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]
    if step_sizes is None:
        pass
    else:
        step_sizes_train = step_sizes[:to_train]
        step_sizes_test = step_sizes[to_train:]

    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]
        if step_sizes is None:
            pass
        else:
            step_sizes_test = step_sizes_test[:-1 * to_drop]
    if step_sizes is None:
        return (x_train, y_train), (x_test, y_test)
    else:
        if step_sizes_train.shape[-1] != 1:
            step_sizes_train = np.expand_dims(step_sizes_train, -1)
            step_sizes_test = np.expand_dims(step_sizes_test, -1)
        return (x_train, y_train, step_sizes_train), (x_test, y_test, step_sizes_test)


def prep_dataframe(datadf, steps, input_cols, output_cols):

    input_data = np.empty(shape=[0, steps, len(input_cols)])
    output_data = np.empty(shape=[0, len(output_cols)])
    step_size_data = np.empty(shape=[0, steps])

    for i, grp in datadf.groupby('file_number'):
        step_sizes = grp['t'].diff().values[1:]

        rollx, rolly, rollsteps = rolling_window(
            grp.loc[grp.index[:-1], input_cols].values,
            grp.loc[grp.index[1:], output_cols].values,
            steps, step_sizes)

        input_data = np.append(input_data, rollx, axis=0)
        output_data = np.append(output_data, rolly, axis=0)
        step_size_data = np.append(step_size_data, rollsteps, axis=0)


    step_size_data = np.reshape(
        step_size_data, (step_size_data.shape[0], step_size_data.shape[1], 1))

    return (input_data, output_data, step_size_data)
