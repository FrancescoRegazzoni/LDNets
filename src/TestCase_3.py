#!/usr/bin/env python3
#%% Import modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

import utils
import optimization

#%% Set some hyperparameters
dt = 1
dt_base = 205
num_latent_states = 12

#%% Define problem
problem = {
    "space": {
        "dimension" : 1
    },
    "input_parameters": [ ],
    "input_signals": [
        { "name": "Iapp1" },
        { "name": "Iapp2" }
    ],
    "output_fields": [
        { "name": "u" }
    ]
}

normalization = {
    'space': { 'min' : [0.0], 'max' : [100.0]},
    'time': { 'time_constant' : dt_base},
    'input_signals': {
        'Iapp1': { 'min': -2.5, 'max': 2.5},
        'Iapp2': { 'min': -2.5, 'max': 2.5},
    },
    'output_fields': {
        'u': { 'min': 0.0, "max": 1.2 }
    }
}

#%% Dataset
data_set_path = '../data/AP1D'

dataset_train = utils.AP_create_dataset(data_set_path, np.arange(  0, 100))
dataset_valid = utils.AP_create_dataset(data_set_path, np.arange(100, 200))
dataset_tests = utils.AP_create_dataset(data_set_path, np.arange(200, 400))

# For reproducibility (delete if you want to test other random initializations)
np.random.seed(0)
tf.random.set_seed(0)

# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset(dataset_train, problem, normalization, dt = dt, num_points_subsample = 20)
utils.process_dataset(dataset_valid, problem, normalization, dt = dt, num_points_subsample = 20)
utils.process_dataset(dataset_tests, problem, normalization, dt = dt)

#%% Model construction
# dynamics network
input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)
NNdyn_base = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(num_latent_states)
        ])
NNdyn_base.summary()
# enforcing equilibrium condition
inp_eq = np.zeros((1,num_latent_states + len(problem['input_signals'])))
NNdyn = lambda inp: NNdyn_base(inp) - NNdyn_base(inp_eq)

# reconstruction network
input_shape = (None, None, num_latent_states + problem['space']['dimension'])
NNrec = tf.keras.Sequential([
            tf.keras.layers.Dense(17, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(17, activation = tf.nn.tanh),
            tf.keras.layers.Dense(17, activation = tf.nn.tanh),
            tf.keras.layers.Dense(17, activation = tf.nn.tanh),
            tf.keras.layers.Dense(17, activation = tf.nn.tanh),
            tf.keras.layers.Dense(len(problem['output_fields']))
        ])
NNrec.summary()

def evolve_dynamics(dataset):
    # intial condition
    state = tf.zeros((dataset['num_samples'], num_latent_states), dtype=tf.float64)
    state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    state_history = state_history.write(0, state)
    dt_ref = normalization['time']['time_constant']
    
    # time integration
    for i in tf.range(dataset['num_times'] - 1):
        state = state + dt/dt_ref * NNdyn(tf.concat([state, dataset['inp_signals'][:,i,:]], axis = -1))
        state_history = state_history.write(i + 1, state)

    return tf.transpose(state_history.stack(), perm=(1,0,2))

def reconstruct_output(dataset, states):
    states_expanded = tf.broadcast_to(tf.expand_dims(states, axis = 2), 
        [dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states])
    return NNrec(tf.concat([states_expanded, dataset['points_full']], axis = 3))

def LDNet(dataset):
    states = evolve_dynamics(dataset)
    return reconstruct_output(dataset, states)

# Loss function
def MSE(dataset):
    out_fields = LDNet(dataset)
    error = out_fields - dataset['out_fields']
    return tf.reduce_mean(tf.square(error))

def weights_reg(NN):
    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

alpha_reg = 4.7e-3
def loss(): return MSE(dataset_train) + alpha_reg * (weights_reg(NNdyn_base) + weights_reg(NNrec))
def MSE_valid(): return MSE(dataset_valid)

#%% Training
trainable_variables = NNdyn_base.variables + NNrec.variables
opt = optimization.OptimizationProblem(trainable_variables, loss, MSE_valid)

num_epochs_Adam = 200
num_epochs_BFGS = 5000

print('training (Adam)...')
opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
print('training (BFGS)...')
opt.optimize_BFGS(num_epochs_BFGS)

fig, axs = plt.subplots(1, 1)
axs.loglog(opt.iterations_history, opt.loss_train_history, 'o-', label = 'training loss')
axs.loglog(opt.iterations_history, opt.loss_valid_history, 'o-', label = 'validation loss')
axs.axvline(num_epochs_Adam)
axs.set_xlabel('epochs'), plt.ylabel('loss')
axs.legend()

#%% Testing
# Compute predictions.
out_fields = LDNet(dataset_tests)

# Since the models work with normalized data, we map back the outputs into the original ranges.
out_fields_FOM = utils.denormalize_output(dataset_tests['out_fields'], problem, normalization).numpy()
out_fields_ROM = utils.denormalize_output(out_fields                 , problem, normalization).numpy()

NRMSE = np.sqrt(np.mean(np.square(out_fields_ROM - out_fields_FOM))) / (np.max(out_fields_FOM) - np.min(out_fields_FOM))

import scipy.stats
R_coeff = scipy.stats.pearsonr(np.reshape(out_fields_ROM, (-1,)), np.reshape(out_fields_FOM, (-1,)))

print('Normalized RMSE:       %1.3e' % NRMSE)
print('Pearson dissimilarity: %1.3e' % (1 - R_coeff[0]))

#%% Postprocessing
fig = utils.plot_output_1D(dataset_tests, out_fields_FOM, out_fields_ROM, 5, 4)
fig.savefig('TestCase3.png')