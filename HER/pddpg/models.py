import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, discrete_action_size, cts_action_size, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.discrete_action_size = discrete_action_size
        self.cts_action_size = cts_action_size
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False, num_layers=3, hidden_units=64):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            for _ in range(num_layers):
                x = tf.layers.dense(x, hidden_units)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
             
            x_discrete = tf.layers.dense(x, self.discrete_action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x_cts = tf.layers.dense(x, self.cts_action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            
            x = tf.concat(values=[x_discrete, x_cts], axis=-1)
            ## for the sake of quick recalling
            y = tf.identity(x, name="preactivation")
            
            x = tf.nn.tanh(y)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False, num_layers=3, hidden_units=64):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, hidden_units)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)

            for _ in range(num_layers-1):
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
