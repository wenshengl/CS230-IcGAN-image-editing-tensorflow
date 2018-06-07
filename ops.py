import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

#the implements of leakyRelu
def lrelu(x , alpha = 0.01 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

def conv2d(x, W , b , strides=2, padding_ = 'SAME'):

    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x , W , strides=[1, strides , strides, 1], padding= padding_)
    x = tf.nn.bias_add(x, b)

    return x

def de_conv(x , W , b , out_shape, s = [1,2,2,1], padding_ = 'SAME'):

    '''
    tf.name_scope: name, default_name = None, values = None
    - validates that the given "values" are from the same graph,
    makes that graph the default graph, and pushes a name in that graph
    '''
    with tf.name_scope('deconv') as scope:
        deconv = tf.nn.conv2d_transpose(x , W ,
        out_shape , strides = s , padding= padding_, name=None)
        out = tf.nn.bias_add(deconv , b)
        return out

def fully_connect(x , weight , bias):
    return tf.add(tf.matmul(x , weight) , bias)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    # axis = 3: feature map
    return tf.concat([x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])], 3)

    '''
    decay: Decay for the moving average
    scale: if True, multiply by gamma
    scope: variable_scope
    reuse: whether or not the layer and its variables should be reused => scope = scope
    updates_collections: if None, a control dependency would be added to make sure
                        the updates are computed in place.
    '''
def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse = tf.AUTO_REUSE , updates_collections=None)


# #VAE
def conv2d_vae(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def de_conv_vae(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:

            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:

            return deconv, w, biases

        else:

            return deconv

def fully_connect_vae(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):

    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))

    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:

      return tf.matmul(input_, matrix) + bias

# def conv_cond_concat(x, y):
#     """Concatenate conditioning vector on feature map axis."""
#     x_shapes = x.get_shape()
#     y_shapes = y.get_shape()

#     return tf.concat(3 , [x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])])