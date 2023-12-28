import numpy as np
import tensorflow as tf
import scipy.optimize as sopt

class OptimizationProblem():
    """
    Class representing an optimization problem.
    """
    def __init__(self, variables, loss_train, loss_valid):
        """
        Parameters
        ----------
        variables
            List of trainable variables.
        loss_train
            Callable returning the loss function on the training dataset.
        loss_valid
            Callable returning the loss function on the validation dataset.
        """
        self.variables = variables if isinstance(variables, (list, tuple)) else [variables]
        self.loss_train = loss_train
        self.loss_valid = loss_valid
        self.stitcher = VariablesStitcher(self.variables)

        self.compile()

        self.iteration = 0
        self.iterations_history = list()
        self.loss_train_history = list()
        self.loss_valid_history = list()
        self.iteration_callback()

    def compile(self):
        self.ag_train_loss = tf.function(self.loss_train)
        self.ag_train_grad = tf.function(self.compute_gradient)
        self.ag_train_loss_grad = tf.function(lambda params: self.get_gradient_and_loss(params_1d = params))
        self.ag_valid_loss = tf.function(self.loss_valid)

        print('Tracing functions with autograph...')
        self.ag_train_loss()
        self.ag_train_grad()
        self.ag_train_loss_grad(self.stitcher.stitch(self.variables).numpy())
        self.ag_valid_loss()
        print('Tracing completed.')
        
    def get_gradient_and_loss(self, params_1d):
        self.stitcher.update_variables(params_1d)
        with tf.GradientTape(watch_accessed_variables = False) as tape:
            tape.watch(self.variables)
            loss_value = self.loss_train()
        grads = self.stitcher.stitch(tape.gradient(loss_value, self.variables, unconnected_gradients = tf.UnconnectedGradients.ZERO))
        return loss_value, grads
        
    def ag_train_loss_grad_numpy(self, params_1d):
        loss, grad = self.ag_train_loss_grad(params_1d)
        return loss.numpy(), grad.numpy()
        
    def compute_gradient(self):
        with tf.GradientTape(watch_accessed_variables = False) as tape:
            tape.watch(self.variables)
            loss_value = self.loss_train()
        return tape.gradient(loss_value, self.variables, unconnected_gradients = tf.UnconnectedGradients.ZERO)

    def iteration_callback(self):
        if self.iteration % 10 == 0:
            self.iterations_history.append(self.iteration)
            self.loss_train_history.append(self.ag_train_loss())
            self.loss_valid_history.append(self.ag_valid_loss())
            print('epoch% 5d   -   training loss: %1.3e   -   validation loss %1.3e' % 
                  (self.iteration, self.loss_train_history[-1], self.loss_valid_history[-1]))
        self.iteration += 1
        
    def optimize_keras(self, num_epochs, optimizer):
        for _ in range(num_epochs):
            optimizer.apply_gradients(zip(self.ag_train_grad(), self.variables))
            self.iteration_callback()
        
    def optimize_BFGS(self, num_epochs):
        options = {'maxiter': num_epochs, 'gtol': 1e-100}
        init_params = self.stitcher.stitch(self.variables).numpy()

        def callback(_):
            self.iteration_callback()
            return False
    
        sopt.minimize(fun = self.ag_train_loss_grad_numpy,
                x0 = init_params,
                method = 'BFGS',
                jac = True,
                tol = 1e-100,
                options = options,
                callback = callback)

class VariablesStitcher:
    """
    Helper class to reshape a list of tf.Variable's into a 1D tf.Tensor/np.array and vice versa.
    """
    def __init__(self, variables):

        self.variables = variables

        # obtain the shapes of the variables
        self.shapes = tf.shape_n(self.variables)
        self.n_tensors = len(self.shapes)

        count = 0
        self.idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        self.__num_variables = count

        self.part = tf.constant(part)

    @property
    def num_variables(self):
        return self.__num_variables

    def update_variables(self, params_1d):
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.variables[i].assign(tf.reshape(param, shape))

    def reverse_stitch(self, params_1d):
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        return [tf.reshape(param, shape) for i, (shape, param) in enumerate(zip(self.shapes, params))]

    def stitch(self, v = None):
        return tf.dynamic_stitch(self.idx, self.variables if v is None else v)