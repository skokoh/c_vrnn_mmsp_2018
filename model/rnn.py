import tensorflow as tf
from util.tensorflow import rnn

def gru(inputs, size=512, initial_state=None, name="gru", reuse=False, dtype=tf.float32, keep_prob=1.0):
    with tf.variable_scope(name, reuse=reuse) as scope:
        cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.GRUCell(size, reuse=reuse, activation=tf.nn.selu), output_keep_prob=keep_prob)
        input_shape = tf.shape(inputs)

        if initial_state is None:
            initial_state = cell.zero_state(batch_size=input_shape[0], dtype=dtype)

        outputs,state = tf.nn.dynamic_rnn(cell, inputs, 
                            initial_state=initial_state, dtype=dtype)
        return outputs, state
    
def conditional_gru(inputs, size=512, keep_prob=1.0, name="conditional_gru", initial_state=None, reuse=False, dtype=tf.float32):
    with tf.variable_scope(name, reuse=reuse):
        cell = rnn.ConditionalGRU(size, reuse=tf.get_variable_scope().reuse, activation=tf.nn.selu)
        input_shape = tf.shape(inputs)
        
        if initial_state is None:
            initial_state = cell.zero_state(batch_size=input_shape[0], dtype=dtype)
            
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=dtype)
        
        return outputs, state

def lstm(inputs, size=512, initial_state=None, name="lstm", state_is_tuple=True, reuse=False, dtype=tf.float32, keep_prob=1.0):
    with tf.variable_scope(name, reuse=reuse) as scope:
        cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(size, reuse=reuse, state_is_tuple=state_is_tuple), output_keep_prob=keep_prob)
        input_shape = tf.shape(inputs)

        if initial_state is None:
            initial_state = cell.zero_state(batch_size=input_shape[0], dtype=dtype)

        outputs,state = tf.nn.dynamic_rnn(cell, inputs, 
                            initial_state=initial_state, dtype=dtype)
        return outputs, state

def bidirectional_gru(inputs, layers=1, size=512, keep_prob=1.0, name="conditional_gru", initial_state=None, reuse=False, dtype=tf.float32):
    with tf.variable_scope(name, reuse=reuse):
        cell_fw = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.GRUCell(size, reuse=tf.get_variable_scope().reuse, activation=tf.nn.selu), output_keep_prob=keep_prob) for _ in range(layers)]
        cell_bw = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.GRUCell(size, reuse=tf.get_variable_scope().reuse, activation=tf.nn.selu), output_keep_prob=keep_prob) for _ in range(layers)]
        input_shape = tf.shape(inputs)
        
        if initial_state is None:
            initial_state = layers*[cell_fw[0].zero_state(batch_size=input_shape[0], dtype=dtype)]
            
        outputs, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, initial_states_fw=initial_state, initial_states_bw=initial_state, dtype=dtype)

        #outputs = tf.concat(outputs, axis=2)
        
        return outputs, state_fw, state_bw

def bidirectional_conditional_gru(inputs, size=512, keep_prob=1.0, name="conditional_gru", initial_state=None, reuse=False, dtype=tf.float32):
    with tf.variable_scope(name, reuse=reuse):
        cell_fw = rnn.ConditionalGRU(size, reuse=tf.get_variable_scope().reuse, activation=tf.nn.selu)
        cell_bw = rnn.ConditionalGRU(size, reuse=tf.get_variable_scope().reuse, activation=tf.nn.selu)
        input_shape = tf.shape(inputs)
        
        if initial_state is None:
            initial_state = cell_fw.zero_state(batch_size=input_shape[0], dtype=dtype)
            
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, initial_state_fw=initial_state, initial_state_bw=initial_state, dtype=dtype)

        outputs = tf.concat(outputs, axis=2)
        
        return outputs, state

def bidirectional_lstm(inputs, size=512, layers=1, keep_prob=1.0, name="conditional_gru", initial_state=None, state_is_tuple=True, reuse=False, dtype=tf.float32):
    with tf.variable_scope(name, reuse=reuse):
        cell_fw = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(size, reuse=tf.get_variable_scope().reuse, state_is_tuple=state_is_tuple), output_keep_prob=keep_prob) for _ in range(layers)]
        cell_bw = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(size, reuse=tf.get_variable_scope().reuse, state_is_tuple=state_is_tuple), output_keep_prob=keep_prob) for _ in range(layers)]
        input_shape = tf.shape(inputs)
        
        if initial_state is None:
            initial_state = layers*[cell_fw[0].zero_state(batch_size=input_shape[0], dtype=dtype)]
            
        outputs, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, initial_states_fw=initial_state, initial_states_bw=initial_state, dtype=dtype)

        #outputs = tf.concat(outputs, axis=2)
        
        return outputs, state_fw, state_bw
