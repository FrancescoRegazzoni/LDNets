import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import interpolate
import scipy.io

def normalize_forw(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return (2.0*v - v_min - v_max) / (v_max - v_min)

def normalize_back(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return 0.5*(v_min + v_max + (v_max - v_min) * v)

def reshape_min_max(n, v_min, v_max, axis = None):
    if axis is not None:
        shape_min = [1] * n
        shape_max = [1] * n
        shape_min[axis] = len(v_min)
        shape_max[axis] = len(v_max)
        v_min = np.reshape(v_min, shape_min)
        v_max = np.reshape(v_max, shape_max)
    return v_min, v_max
    
def analyze_normalization(problem, normalization_definition):
    normalization = dict()
    normalization['dt_base'] = normalization_definition['time']['time_constant']
    normalization['x_min'] = np.array(normalization_definition['space']['min'])
    normalization['x_max'] = np.array(normalization_definition['space']['max'])
    if len(problem.get('input_parameters', [])) > 0:
        normalization['inp_parameters_min'] = np.array([normalization_definition['input_parameters'][v['name']]['min'] for v in problem['input_parameters']])
        normalization['inp_parameters_max'] = np.array([normalization_definition['input_parameters'][v['name']]['max'] for v in problem['input_parameters']])
    if len(problem.get('input_signals', [])) > 0:
        normalization['inp_signals_min'] = np.array([normalization_definition['input_signals'][v['name']]['min'] for v in problem['input_signals']])
        normalization['inp_signals_max'] = np.array([normalization_definition['input_signals'][v['name']]['max'] for v in problem['input_signals']])
    normalization['out_fields_min'] = np.array([normalization_definition['output_fields'][v['name']]['min'] for v in problem['output_fields']])
    normalization['out_fields_max'] = np.array([normalization_definition['output_fields'][v['name']]['max'] for v in problem['output_fields']])
    return normalization

def dataset_normalize(dataset, problem, normalization_definition):
    normalization = analyze_normalization(problem, normalization_definition)
    dataset['times']              = dataset['times'] / normalization['dt_base']
    dataset['points']             = normalize_forw(dataset['points']        , normalization['x_min']             , normalization['x_max']             , axis = 1)
    dataset['points_full']        = normalize_forw(dataset['points_full']   , normalization['x_min']             , normalization['x_max']             , axis = 3)
    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = normalize_forw(dataset['inp_parameters'], normalization['inp_parameters_min'], normalization['inp_parameters_max'], axis = 1)
    if dataset['inp_signals'] is not None:
        dataset['inp_signals']    = normalize_forw(dataset['inp_signals']   , normalization['inp_signals_min']   , normalization['inp_signals_max']   , axis = 2)
    dataset['out_fields']         = normalize_forw(dataset['out_fields']    , normalization['out_fields_min']    , normalization['out_fields_max']    , axis = 3)
    
def denormalize_output(out_fields, problem, normalization_definition):
    normalization = analyze_normalization(problem, normalization_definition)
    return normalize_back(out_fields , normalization['out_fields_min'], normalization['out_fields_max'], axis = 3)
    
def process_dataset(dataset, problem, normalization_definition, dt = None, num_points_subsample = None):
    if dt is not None:
        times = np.arange(dataset['times'][0], dataset['times'][-1] + dt * 1e-10, step = dt)
        if dataset['inp_signals'] is not None:
            dataset['inp_signals'] = interpolate.interp1d(dataset['times'], dataset['inp_signals'], axis = 1)(times)
        dataset['out_fields'] = interpolate.interp1d(dataset['times'], dataset['out_fields'], axis = 1)(times)
        dataset['times'] = times

    num_samples = dataset['out_fields'].shape[0]
    num_times = dataset['times'].shape[0]
    num_points = dataset['points'].shape[0]
    num_x = dataset['points'].shape[1]

    points_full = np.broadcast_to(dataset['points'][None,None,:,:], [num_samples, num_times, num_points, num_x])
    
    if num_points_subsample is None:
        dataset['points_full'] = points_full
    else:
        idxs = np.array([[np.random.choice(num_points, num_points_subsample) for j in range(num_times)] for i in range(num_samples)])
        dataset['points_full'] = np.array([[points_full          [i,j,idxs[i,j,:],:] for j in range(num_times)] for i in range(num_samples)])
        dataset['out_fields']  = np.array([[dataset['out_fields'][i,j,idxs[i,j,:],:] for j in range(num_times)] for i in range(num_samples)])

    dataset['num_points'] = dataset['points_full'].shape[2]
    dataset['num_times'] = num_times
    dataset['num_samples'] = num_samples

    dataset_normalize(dataset, problem, normalization_definition)

    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = tf.convert_to_tensor(dataset['inp_parameters'], tf.float64)
    if dataset['inp_signals'] is not None:
        dataset['inp_signals'] = tf.convert_to_tensor(dataset['inp_signals'], tf.float64)
    dataset['out_fields'] = tf.convert_to_tensor(dataset['out_fields'], tf.float64)

def plot_output_1D(dataset, out_fields_ref, out_fields_app, n_row, n_col, title_ROM = 'ROM'):
    fig = plt.figure(figsize=(10, 8), constrained_layout=False)
    outer_grid = fig.add_gridspec(n_row, n_col, wspace=1e-1, hspace=3e-1)

    t = dataset['times']
    x = dataset['points']
    X, T = np.meshgrid(x,t)

    vmin, vmax = np.min(out_fields_ref), np.max(out_fields_ref)
    for i_sample in range(n_col*n_row):
        idx_col = i_sample % n_col
        idx_row = int((i_sample - idx_col) / n_col)

        inner_grid = outer_grid[idx_row, idx_col].subgridspec(1, 2, wspace=0, hspace=0)
        axs = inner_grid.subplots()

        axs[0].pcolormesh(X, T, out_fields_ref[i_sample,:,:,0], shading='auto', vmin = vmin, vmax = vmax)
        axs[1].pcolormesh(X, T, out_fields_app[i_sample,:,:,0], shading='auto', vmin = vmin, vmax = vmax)
        axs[0].set_title('FOM')
        axs[1].set_title(title_ROM)

        for ax in axs.flatten():
            ax.set(xticks=[], yticks=[])
            for spine in ['top', 'bottom', 'left', 'right']: ax.spines[spine].set_visible(True)
    return fig

# Data conversion functions for specific Test Cases

def ADR_create_dataset(dataset, idxs):
    new_dataset = {
        'points' : dataset['x'][:, None], # [num_points x num_coordinates]
        'times' : dataset['t'], # [num_times]
        'out_fields' : dataset['output'][idxs,:,:,None], # [num_samples x num_times x num_points x num_fields]
    }
    if 'param' in dataset.keys():
        new_dataset['inp_parameters'] = dataset['param'][idxs,:] # [num_samples x num_par]
    else:
        new_dataset['inp_parameters'] = None
    
    if 'forcing' in dataset.keys():
        new_dataset['inp_signals'] = dataset['forcing'][idxs,:,:] # [num_samples x num_times x num_signals]
    else:
        new_dataset['inp_signals'] = None
    
    return new_dataset

def NS_create_dataset(dataset_path, idxs):
    print('loading dataset %s' % dataset_path)
    dataset = np.load(dataset_path, allow_pickle = True)[()]
    print('loaded dataset')

    X, Y = np.meshgrid(dataset['x'], dataset['y'])
    x = np.reshape(X, (-1,))
    y = np.reshape(Y, (-1,))
    points = np.concatenate([x[:,None], y[:,None]], axis = 1)

    return {
        'points' : points, # [num_points x num_coordinates]
        'times' : dataset['t'], # [num_times]
        'inp_parameters' : None, # [num_samples x num_par]
        'inp_signals' : dataset['velocity_top'][idxs,:,None], # [num_samples x num_times x num_signals]
        'out_fields' : np.concatenate([dataset['ux'][idxs,:,:,None], dataset['uy'][idxs,:,:,None]], axis = 3), # [num_samples x num_times x num_points x num_fields]
    }

def AP_create_dataset(base_path, idxs, points_subsampling_rate = 8, time_steps = 501):

    out_fields_vec = []
    inp_signals_vec = []

    for idx in idxs:
        data_path_sol = base_path + '/APsolution%d.npy' % idx
        data_path_inp = base_path + '/APsetting%d.csv' % idx

        out_fields_vec.append(np.load(data_path_sol))
        dataframe = pd.read_csv(data_path_inp)

        t = np.array(dataframe['t'])
        inp_signals_vec.append(np.array(dataframe[['u(t)_1', 'u(t)_2']]))

        print('loaded sample %d' % idx)
        
    nodes = 801
    domain_length = 100.0
    delta_x = domain_length/(nodes-1)
    x = np.arange(0,domain_length+delta_x,delta_x)

    if time_steps is None:
        time_steps = t.shape[0]

    dataset = {
            'points' : x[::points_subsampling_rate, None], # [num_points x num_coordinates]
            'times' : t[:time_steps], # [num_times]
            'out_fields' : np.array(out_fields_vec)[:,:time_steps,::points_subsampling_rate,None], # [num_samples x num_times x num_points x num_fields]
            'inp_parameters' : None, # [num_samples x num_par]
            'inp_signals' : np.array(inp_signals_vec)[:,:time_steps,:] # [num_samples x num_times x num_signals]
        }
    
    return dataset

def reentry_create_dataset(base_path,first_sample,last_sample):

    N_points  = 3*898
    N_times   = 181
    dim_out   = 1
    dim_points = 2

    N_samples = last_sample - first_sample

    dataset_matrix = np.zeros((N_samples,N_times,N_points,dim_out))
    points_matrix  = np.zeros((N_points,dim_points))
    diameter_vec = np.zeros((N_samples,1))
    impulse_vec  = np.zeros((N_samples,N_times,1)) 

    mat_1 = scipy.io.loadmat(base_path + '/sample_1.mat')

    x = mat_1['data_train']['x']
    x = x[0][0][0][np.arange(0,8081,3)]
    y = mat_1['data_train']['y']
    y = y[0][0][0][np.arange(0,8081,3)]

    time_vec = 450*np.arange(0,1.0001,1.0/180).T

    for k in range(N_samples):

        print('loading dataset %d' % (first_sample + k + 1))
        mat = scipy.io.loadmat(base_path + '/sample_'+str(first_sample + k+1)+'.mat')
        vh = mat['data_train']['vh']
        output = vh[0][0][np.arange(0,8081,3)]
        param = mat['data_train']['param']

        for j in range(N_points):
            for i in range(N_times):
                dataset_matrix[k,i,j,0] = output[j,i]
                timing = param[0][0][2]
                impulse_vec[k,i,0] = (time_vec[i]>=timing[0])*(time_vec[i]<=(timing[0]+2.5))

            points_matrix[j,0] = x[j]
            points_matrix[j,1] = y[j]

        diameter_vec[k,0] = param[0][0][3]

    dataset = {
        'points' : points_matrix , # [num_points x num_coordinates]
        'times' : np.arange(0,1.0001,1.0/180).T , # [num_times]
        'inp_parameters' : diameter_vec, # [num_samples x num_par]
        'inp_signals' : impulse_vec, # [num_samples x num_times x num_signals]
        'out_fields' : dataset_matrix, # [num_samples x num_times x num_points x num_fields]
    }
    
    return dataset