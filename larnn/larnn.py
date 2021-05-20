
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer


class LinearAntisymmetricCell(Layer):
    '''
        Implements the Linear Antisymmetric RNN cell, which can learn the parameters
        of discretized versions of the linear ODE
            h_dot = Wh*h + f(Vx + b)
        The LARNN structure is based on numeric solutions to the above ODE,
        solved using either the forward Euler, backward Euler or implicit midpoint method.

        The LinearAntisymmetricCell is implemented as a RNN cell as specified in the
        Tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

        When called, the cell expects the input at the current step, and the
        current cell state. The expected input depends on whether the step_size varies at each
        step or not, as indicated by the 'var_step_size' argument. If 'var_step_size' is
        False, the cell expects a single input of size (batch_size, nfeatures), while if
        'var_step_size' is True, it expects an input tuple with two entries:
        (input_data, epsilons) of sizes ((batch_size, nfeatures), (batch_size, 1)).
        
        Parameters
        ----------
        units : int
            Number of cells to add in paralell.
        step_size : float
            The solver step size. Ignored if 'var_step_size' is true.
            If 'trainable_step_size' is true, this is by default the initial step_size.
        var_step_size : bool
            If True, the step_size may vary at each processing step. Requires an additional
            input of step_sizes in the cell call.
        trainable_step_size : bool
            If True, 'step_size' is set to be a trainable parameter.
        method : str
            Accepted values are 'forward', 'backward' and 'midpoint', which corresponds
            to solving the linear ODE with the forward Euler method, backward Euler
            method or the implicit midpoint rule.
        activation : str
            Non-linearity to use for modeling the input non-linearity. String abreviations
            for all Keras activation functions are accepted.
        use_bias : bool
            Add a bias term.
        kernel_initializer : str
            tf.keras initializer.
        recurrent_initializer : str
            tf.keras initializer.
        bias_initializer : str
            tf.keras initializer.
        step_size_initializer : str
            tf.keras initializer.
        kernel_regularizer : str
            tf.keras regularizer.
        recurrent_regularizer : str
            tf.keras regularizer.
        bias_regularizer : str
            tf.keras regularizer.
        step_size_regularizer : str
            tf.keras regularizer.
        kernel_constraint : str
            tf.keras constraint.
        recurrent_constraint : str
            tf.keras constraint.
        step_size_constraint : str
            tf.keras constraint.
        bias_constraint : str
            tf.keras constraint.

    '''

    def __init__(self, units,
                 step_size=0.1,
                 var_step_size=False,
                 trainable_step_size=False,
                 method='midpoint',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='uniform',
                 recurrent_initializer='uniform',
                 bias_initializer='uniform',
                 step_size_initializer='constant',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 step_size_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 step_size_constraint=constraints.NonNeg(),
                 bias_constraint=None,
                 **kwargs):

        super(LinearAntisymmetricCell, self).__init__(**kwargs)

        methods = ('forward', 'backward', 'midpoint')

        if method not in methods:
            raise ValueError(f'Unknown ODE solver method "{method}".'
                             + f' Supported methods: {methods}')
        if (not var_step_size) and (not step_size):
            raise ValueError('Either "step_size" must be set to a positive'
                             ' number, or "var_step_size" must be True')
        if var_step_size and trainable_step_size:
            raise ValueError('"var_step_size" and "trainable_step_size" cannot both be True.')
        if not var_step_size:
            if step_size <= 0:
                raise ValueError('Invalid value for "step_size". "step_size" > 0 required.')

        self.trainable_step_size = trainable_step_size
        self.var_step_size = var_step_size
        self.step_size = float(step_size)

        self.units = units
        self.activation = activations.get(activation)
        self.method = method

        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        if step_size_initializer == 'constant':
            self.step_size_initializer = initializers.Constant(value=self.step_size)
        else:
            self.step_size_initializer = initializers.get(step_size_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.step_size_regularizer = regularizers.get(step_size_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.step_size_constraint = constraints.get(step_size_constraint)

        self.state_size = units
        self.output_size = units

    def build(self, input_shapes):
        if isinstance(input_shapes[0], (list, tuple)):
            input_shape = input_shapes[0]
        else:
            input_shape = input_shapes

        self.kernel = self.add_weight(
            name="V",
            shape=(input_shape[-1], self.output_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)

        self.recurrent_kernel = self.add_weight(
            name="W",
            shape=(self.units, self.units),
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            trainable=True)

        if self.trainable_step_size:
            self.trainstep = self.add_weight(
                name="step_size",
                shape=(1,),
                initializer=self.step_size_initializer,
                regularizer=self.step_size_regularizer,
                constraint=self.step_size_constraint,
                trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, states):
        h_prev = states[0]
        if self.var_step_size:
            inputs, epsilon = tf.nest.flatten(inputs)
            assert (len(epsilon.shape) == 2 and epsilon.shape[-1] == 1), f'Epsilon must be of shape (batch_size, 1)'
        elif self.trainable_step_size:
            # Ensure epsilon > 0
            epsilon = self.trainstep + 1e-11
        else:
            epsilon = self.step_size

        W = self.recurrent_kernel
        W_h = tf.transpose(W) - W
        Vx = tf.matmul(inputs, self.kernel)

        if self.use_bias:
            Vx = K.bias_add(Vx, self.bias, data_format='channels_last')

        act_term = Vx
        if self.activation:
            act_term = self.activation(act_term)

        if self.method == 'forward':
            # Forward Euler step
            h_new = h_prev + tf.multiply(epsilon, tf.matmul(h_prev, W_h) + act_term)
        else:
            Iden = tf.eye(self.units)
            if self.var_step_size:
                epsmat = tf.expand_dims(epsilon, -1)
                epsmat = tf.tile(epsmat, tf.constant([1, 1, self.state_size]))
            else:
                epsmat = epsilon
            if self.method == 'midpoint':
                # Implicit midpoint rule step
                if self.var_step_size:
                    LHS_matrix = Iden - 0.5*tf.multiply(epsmat, tf.transpose(W_h))
                else:
                    LHS_matrix = Iden - 0.5*tf.multiply(epsmat, W_h)
                RHS = (h_prev + 0.5*tf.multiply(epsilon, tf.matmul(h_prev, W_h))
                       + tf.multiply(epsilon, act_term))

            elif self.method == 'backward':
                # Backward Euler step
                if self.var_step_size:
                    LHS_matrix = Iden - tf.multiply(epsmat, tf.transpose(W_h))
                else:
                    LHS_matrix = Iden - tf.multiply(epsmat, W_h)
                RHS = h_prev + tf.multiply(epsilon, act_term)

            if self.var_step_size:
                RHS = tf.expand_dims(RHS, -1)
                h_new = tf.linalg.solve(LHS_matrix, RHS)
                h_new = tf.squeeze(h_new, -1)
            else:
                LHS_matrix = tf.transpose(LHS_matrix)
                RHS = tf.transpose(RHS)
                h_new = tf.transpose(tf.linalg.solve(LHS_matrix, RHS))

        return h_new, [h_new]

    def get_config(self):
        config = {'units': self.units,
                  'method': self.method,
                  'activation': activations.serialize(self.activation),
                  'step_size': self.step_size,
                  'var_step_size': self.var_step_size,
                  'use_bias': self.use_bias,
                  'trainable_step_size': self.trainable_step_size,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'step_size_initializer': initializers.serialize(self.step_size_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'step_size_regularizer': regularizers.serialize(self.step_size_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'step_size_constraint': constraints.serialize(self.step_size_constraint)}
        base_config = super(LinearAntisymmetricCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
