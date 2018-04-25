import tensorflow as tf

def conv_relu_pool(inputs, kernel_shape, pool_shape, name="conv2d", reuse=False, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name) as scope:
        conv_kernel = tf.get_variable('conv2d', shape=kernel_shape, initializer=initializer, dtype=dtype)
        conv_bias = tf.get_variable('conv2d_b', shape=kernel_shape[3], dtype=dtype)
        
        conv = tf.nn.conv2d(inputs, conv_kernel, [1,1,1,1], "VALID")
        preact = tf.nn.bias_add(conv, conv_bias)
        relu = tf.nn.relu(preact)
        pool = tf.nn.max_pool(relu, pool_shape, pool_shape, "VALID")

        return pool

def conv_selu(inputs, kernel_shape, name="conv2d", reuse=False, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), keep_prob=1.0):
    with tf.variable_scope(name) as scope:
        conv_kernel = tf.get_variable('conv2d', shape=kernel_shape, initializer=initializer, dtype=dtype)
        conv_bias = tf.get_variable('conv2d_b', shape=kernel_shape[3], dtype=dtype)
        
        conv = tf.nn.conv2d(inputs, conv_kernel, [1,1,1,1], "VALID")
        preact = tf.nn.bias_add(conv, conv_bias)
        selu = tf.nn.selu(preact)
        selu = tf.nn.dropout(selu, keep_prob=keep_prob)
        return selu

def conv_selu_pool(inputs, kernel_shape, pool_shape, name="conv2d", reuse=False, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), keep_prob=1.0):
    selu = conv_selu(inputs, kernel_shape, name=name, reuse=reuse, dtype=dtype, initializer=initializer)
    selu = tf.nn.dropout(selu, keep_prob=keep_prob)
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(selu, pool_shape, pool_shape, "VALID")

        return pool
