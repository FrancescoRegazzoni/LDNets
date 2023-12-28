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
dt = 0.2
dt_base = 5.4
num_latent_states = 1

#%% Define problem
problem = {
    "space": {
        "dimension" : 2
    },
    "input_parameters": [],
    "input_signals": [
        { "name": "u" }
    ],
    "output_fields": [
        { "name": "ux" },
        { "name": "uy" }
    ]
}

normalization = {
    'space': {
        'min' : [0, 0],
        'max' : [1, 1],
    },
    'time': {
        'time_constant' : dt_base
    },
    'input_signals': {
        'u': { 'min': -20, 'max': 20 }
    },
    'output_fields': {
        'ux': { 'min': -20, "max": 20 },
        'uy': { 'min': -20, 'max': 20 },
    }
}

#%% Dataset

data_set_1_path = '../data/NS/T20_80samples.npy'
data_set_2_path = '../data/NS/T20_20samples.npy'
data_set_3_path = '../data/NS/T40_10samples.npy'

dataset_train = utils.NS_create_dataset(data_set_1_path, np.arange(0, 80))
dataset_valid = utils.NS_create_dataset(data_set_2_path, np.arange(0, 20)) 
dataset_tests = utils.NS_create_dataset(data_set_3_path, np.arange(0, 10))

# For reproducibility (delete if you want to test other random initializations)
np.random.seed(0)
tf.random.set_seed(0)

# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset(dataset_train, problem, normalization, dt = dt, num_points_subsample = 200)
utils.process_dataset(dataset_valid, problem, normalization, dt = dt, num_points_subsample = 200)
utils.process_dataset(dataset_tests, problem, normalization, dt = dt)

#%% Define LDNet model

# dynamics network
input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)
NNdyn = tf.keras.Sequential([
            tf.keras.layers.Dense(7, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(7, activation = tf.nn.tanh),
            tf.keras.layers.Dense(num_latent_states)
        ])
NNdyn.summary()

# reconstruction network
input_shape = (None, None, num_latent_states + len(problem['input_signals']) + problem['space']['dimension'])
NNrec = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(24, activation = tf.nn.tanh),
            tf.keras.layers.Dense(24, activation = tf.nn.tanh),
            tf.keras.layers.Dense(24, activation = tf.nn.tanh),
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
    inp_signals_expanded = tf.broadcast_to(tf.expand_dims(dataset['inp_signals'], axis = 2),
        [dataset['num_samples'], dataset['num_times'], dataset['num_points'], len(problem['input_signals'])])
    output = NNrec(tf.concat([states_expanded, inp_signals_expanded, dataset['points_full']], axis = 3))
    # nonlinear transformation to compress long tails
    alpha = 0.05
    output = (output**3 + alpha*output)/(1+alpha)
    return output

def LDNet(dataset):
    states = evolve_dynamics(dataset)
    return reconstruct_output(dataset, states)

#%% Loss function
weight_direction = 0.1
epsilon = 1e-4

def get_direction(velocity): 
    return tf.math.divide(velocity, (epsilon + tf.expand_dims(tf.norm(velocity, axis = 3), axis = -1)))

def loss(dataset, target_velocity, target_direction):
    velocity = LDNet(dataset)
    MSE_velocity = tf.reduce_mean(tf.square(velocity - target_velocity))
    direction = get_direction(velocity)
    MSE_direction = tf.reduce_mean(tf.square(direction - target_direction))
    return MSE_velocity + weight_direction * MSE_direction

target_direction_train = get_direction(dataset_train['out_fields'])
target_direction_valid = get_direction(dataset_valid['out_fields'])
loss_train = lambda: loss(dataset_train, dataset_train['out_fields'], target_direction_train)
loss_valid = lambda: loss(dataset_valid, dataset_valid['out_fields'], target_direction_valid)

#%% Training
trainable_variables = NNdyn.variables + NNrec.variables
opt = optimization.OptimizationProblem(trainable_variables, loss_train, loss_valid)

num_epochs_Adam = 200
num_epochs_BFGS = 10000

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

# Since the LDNet works with normalized data, we map back the outputs into the original ranges.
out_fields_FOM = utils.denormalize_output(dataset_tests['out_fields'], problem, normalization).numpy()
out_fields_ROM = utils.denormalize_output(out_fields                 , problem, normalization).numpy()

NRMSE = np.sqrt(np.mean(np.square(out_fields_ROM - out_fields_FOM))) / (np.max(out_fields_FOM) - np.min(out_fields_FOM))

import scipy.stats
R_coeff = scipy.stats.pearsonr(np.reshape(out_fields_ROM, (-1,)), np.reshape(out_fields_FOM, (-1,)))

print('Normalized RMSE:       %1.3e' % NRMSE)
print('Pearson dissimilarity: %1.3e' % (1 - R_coeff[0]))

#%% Postprocessing
num_times = 8
i_sample = 0

n_pts = int(np.sqrt(dataset_tests['points'].shape[0]))
X = np.reshape(dataset_tests['points'][:,0], (n_pts,n_pts))
Y = np.reshape(dataset_tests['points'][:,1], (n_pts,n_pts))

v_min = np.min(out_fields_FOM[i_sample,:,:,:], axis = (0,1))
v_max = np.max(out_fields_FOM[i_sample,:,:,:], axis = (0,1))

times = np.linspace(0, len(dataset_tests['times']) - 1, num = num_times, dtype = int)
fig, axs = plt.subplots(4, num_times, figsize = (2*num_times, 8))
for idxT, iT in enumerate(times):
    axs[0, idxT].set_title('t = %.2f' % (dataset_tests['times'][iT] * dt_base))
    for i in range(2):
        levels = matplotlib.ticker.MaxNLocator(nbins=40).tick_values(v_min[i], v_max[i])
        Z_FOM = np.reshape(out_fields_FOM[i_sample,iT,:,i], (n_pts,n_pts))
        Z_ROM = np.reshape(out_fields_ROM[i_sample,iT,:,i], (n_pts,n_pts))
        axs[2*i+0, idxT].contourf(X, Y, Z_FOM, cmap='magma', levels=levels, extend = 'both')    
        axs[2*i+1, idxT].contourf(X, Y, Z_ROM, cmap='magma', levels=levels, extend = 'both')

axs[0, 0].set_ylabel('$v_x$ (FOM)')
axs[1, 0].set_ylabel('$v_x$ (ROM)')
axs[2, 0].set_ylabel('$v_y$ (FOM)')
axs[3, 0].set_ylabel('$v_y$ (ROM)')

for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig('TestCase2.png')