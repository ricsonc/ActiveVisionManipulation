import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.slim as slim
from ipdb import set_trace as st

#the action order is... 2 tr, 2 rt

def convnet(input):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding="SAME",
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': True, 
                                           'epsilon': 1e-5,
                                           'decay': 0.9,
                                           'scale': True,
                                           'updates_collections': None}):

        net = input #128^2
        net = slim.conv2d(net, 16, [4, 4], stride = 2, scope = 'conv1') #64
        net = slim.conv2d(net, 32, [4, 4], stride = 2, scope = 'conv2') #32
        net = slim.conv2d(net, 64, [4, 4], stride = 4, scope = 'conv3') #8
        net = tf.reduce_mean(net, axis = [1,2])
        net = slim.fully_connected(net, 64, activation_fn = None)
    return net


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

    def combine_depth_and_feats(self, depth, feats, reuse = False):
        with tf.variable_scope(self.name, reuse = reuse) as scope:
            depth_embed = convnet(depth)

        if False:
            return tf.concat([feats, depth_embed], axis = -1)
        else:
            return feats + depth_embed

    def zeroifstatic(self, x):
        if self.static:
            return tf.concat([x[:,:2], tf.zeros_like(x[:,2:])], axis = 1)
        else:
            return x
        
class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, static = False):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.static = static
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
             
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            
            ## for the sake of quick recalling
            y = tf.identity(x, name="preactivation")
            x = tf.nn.tanh(y)

            x = self.zeroifstatic(x)

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


class DepthActor(Model):
    def __init__(self, nb_actions, name='actor', static = False, **kwargs):
        super(DepthActor, self).__init__(name=name)
        self.static = static
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False, num_layers=3, hidden_units=64, **kwargs): 

        with tf.variable_scope(self.name, reuse = reuse) as scope:        
            x, depth = obs

            for _ in range(num_layers):
                x = tf.layers.dense(x, hidden_units)
                x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
        
            x = self.combine_depth_and_feats(depth, x, reuse = reuse)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            ## for the sake of quick recalling
            y = tf.identity(x, name="preactivation")
            x = tf.nn.tanh(y)

            x = self.zeroifstatic(x)
        return x
    
class DepthCritic(Model):
    def __init__(self, name='critic', **kwargs):
        super(DepthCritic, self).__init__(name=name)

    def __call__(self, obs, action, reuse=False, num_layers=3, hidden_units=64, **kwargs):
        
        with tf.variable_scope(self.name, reuse = reuse) as scope:
            
            x, depth = obs

            x = tf.layers.dense(x, hidden_units)
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            action = tf.layers.dense(action, hidden_units)
            action = tc.layers.layer_norm(action, center=True, scale=True)
            action = tf.nn.relu(action)
            
            x = self.combine_depth_and_feats(depth, x, reuse = reuse)
            x = tf.concat([x, action], axis=-1)
            
            for _ in range(num_layers-1):
                x = tf.layers.dense(x, 64)
                x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

class IgnoreDepthActor(DepthActor):
    def combine_depth_and_feats(self, depth, feats, reuse = False):
        return feats

class IgnoreDepthCritic(DepthCritic):
    def combine_depth_and_feats(self, depth, feats, reuse = False):
        return feats

#only feed rgb to the camera action
#the non-camera action should take only the xyz state
#note that the first two are gripper, rest are action

class FactoredDepthActor(Model):
    def __init__(self, nb_actions, name='actor', static = False, **kwargs):
        super(FactoredDepthActor, self).__init__(name=name)
        self.static = static
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False, num_layers=3, hidden_units=64, **kwargs): 

        with tf.variable_scope(self.name, reuse = reuse) as scope:        
            x, depth = obs

            for _ in range(num_layers):
                x = tf.layers.dense(x, hidden_units)
                x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
        
            x_combined = self.combine_depth_and_feats(depth, x, reuse = reuse)

            x = tf.layers.dense(x, 2, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x_combined = tf.layers.dense(x_combined, self.nb_actions - 2, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

            x = tf.concat([x, x_combined], axis = -1)
            
            ## for the sake of quick recalling
            y = tf.identity(x, name="preactivation")
            x = tf.nn.tanh(y)

            x = self.zeroifstatic(x)
        return x

class ImgDepthActor(Model):
    def __init__(self, nb_actions, name = 'actor', static = False, **kwargs):
        super(ImgDepthActor, self).__init__(name = name)
        self.static = static
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False, num_layers=3, hidden_units=64, **kwargs): 

        with tf.variable_scope(self.name, reuse = reuse) as scope:        
            x, depth = obs

            for _ in range(num_layers):
                x = tf.layers.dense(x, hidden_units)
                x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
        
            x = self.combine_depth_and_feats(depth, x, reuse = reuse)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            ## for the sake of quick recalling
            y = tf.identity(x, name="preactivation")
            x = tf.nn.tanh(y)

            x = self.zeroifstatic(x)
        return x

