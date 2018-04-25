import tensorflow as tf
from util.tensorflow import rnn
from model.cnn import conv_selu, conv_selu_pool
from model.rnn import gru, bidirectional_gru, bidirectional_conditional_gru
from model.rnn import lstm, bidirectional_lstm

def midi_latent_feature(X, z_dim=250, keep_prob=1.0, name='latent_feauture', reuse=False, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name, reuse=reuse) as scope:
        conv1_1 = conv_selu(X, 
                              kernel_shape=[2,2,1,64],
                              name="conv1_1",
                              initializer=initializer, 
                              keep_prob=keep_prob)

        conv1_2_pool = conv_selu_pool(conv1_1, 
                                   kernel_shape=[2, 2 , 64, 32],
                                   pool_shape=[1, 2, 2, 1],
                                   name="conv1_2_pool",
                                   initializer=initializer, 
                                   keep_prob=keep_prob)

        conv2_1 = conv_selu(conv1_2_pool, 
                              kernel_shape=[2,2,32,32],
                              name="conv2_1",
                              initializer=initializer,
                              keep_prob=keep_prob)

        conv2_2_pool = conv_selu_pool(conv2_1, 
                                   kernel_shape=[2, 2 , 32, 32],
                                   pool_shape=[1, 2, 2, 1],
                                   name="conv2_2_pool",
                                   initializer=initializer,
                                   keep_prob=keep_prob)

        conv3_1 = conv_selu(conv2_2_pool, 
                              kernel_shape=[1,1,32,32],
                              name="conv3_1",
                              initializer=initializer,
                              keep_prob=keep_prob)

        conv3_2_pool = conv_selu_pool(conv3_1, 
                                   kernel_shape=[1, 1 , 32, 32],
                                   pool_shape=[1, 2, 2, 1],
                                   name="conv3_2_pool",
                                   initializer=initializer,
                                   keep_prob=keep_prob)

        h = tf.layers.dense(tf.contrib.layers.flatten(conv3_2_pool), 1000, activation=tf.nn.selu, kernel_initializer=initializer)
    
        h = tf.nn.dropout(h, keep_prob=keep_prob)

        fc2 = tf.layers.dense(h, z_dim, activation=tf.nn.selu, kernel_initializer=initializer)

        fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

        return fc2

def Q(X, z_dim=250, h_dim=512, name_scope="Encoder", keep_prob=1.0, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.name_scope(name_scope):
        with tf.name_scope("Encoder_RNN"):
            #Expand dims for rnn to [1, T, k]
            #rnn_in = tf.expand_dims(X, 0)
            rnn_in = X
            _, rnn_state_fw, rnn_state_bw = bidirectional_gru(rnn_in, size=h_dim, name="lstm_enc", keep_prob=keep_prob)
            rnn_state = tf.concat([rnn_state_fw[0], rnn_state_bw[0]], axis=-1)
             #rnn_state = tf.concat(rnn_state, 1)
            #Reshape
            #rnn_state = tf.reshape(rnn_state[-1,:], [-1, z_dim])
            #rnn_state = tf.nn.dropout(rnn_state, keep_prob=0.7)
        
        with tf.name_scope("Calculate_mu_sigma"):
            Q_W2_mu = tf.get_variable('encoder_mean_weights', shape=[2*z_dim, z_dim], initializer=initializer)
            Q_b2_mu = tf.get_variable('encoder_mean_bias', shape=[z_dim], initializer=tf.zeros_initializer())

            Q_W2_sigma = tf.get_variable('encoder_variance_weights', shape=[2*z_dim, z_dim], initializer=initializer)
            Q_b2_sigma = tf.get_variable('encoder_variance_bias', shape=[z_dim], initializer=tf.zeros_initializer())
            #KL divergence with last GRU state
            z_mu = tf.matmul(rnn_state, Q_W2_mu) + Q_b2_mu
            z_logvar = tf.matmul(rnn_state, Q_W2_sigma) + Q_b2_sigma
        return z_mu, z_logvar

def sample_z(mu, log_var):
    with tf.name_scope("Random_samples"):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

def P(z, X_dim, z_dim=512, h_dim=512, name_scope="Decoder", xt=None, keep_prob=1.0, reuse=False, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.name_scope(name_scope):
        with tf.name_scope("concatenate_noise_and_input"):
            #Expand dims for RNN
            if xt is not None:
                #Used to transform CNN representation
                #xt_shape = xt.get_shape().as_list()
                #xt = tf.reshape(xt, [-1, xt_shape[-1]])
                #P_W1 = tf.get_variable('P_W1', shape=[xt_shape[-1], z_dim], initializer=initializer)
                #P_b1 = tf.get_variable('P_b1', shape=[z_dim], initializer=tf.zeros_initializer())
                #xt_in = tf.matmul(xt, P_W1) + P_b1
                #xt_in = tf.reshape(xt_in, [-1, xt_shape[1], z_dim])
                z = tf.expand_dims(z, axis=1)
                z = tf.concat([z, xt], axis=1)
            else:
                z = tf.expand_dims(z, 0)

        with tf.name_scope("Decoder_RNN"):
            dec_outs, _, _ = bidirectional_gru(z, size=h_dim, name="gru_dec", reuse=reuse, layers=2, keep_prob=keep_prob)
            dec_outs = tf.concat(dec_outs, axis=-1)
            dec_outs = tf.reshape(dec_outs, [-1, 2*h_dim])

            #drop = tf.nn.dropout(dec_outs, keep_prob=keep_prob)

            P_W2 = tf.get_variable('P_W2', shape=[2*h_dim, X_dim], initializer=initializer)
            P_b2 = tf.get_variable('P_b2', shape=[X_dim], initializer=tf.zeros_initializer())

        logits = tf.matmul(dec_outs, P_W2) + P_b2
        prob = tf.nn.sigmoid(logits)
        return prob, logits

def P_conditional(z, X_dim, xt=None, z_dim=512, h_dim=512, name_scope="Decoder",  keep_prob=1.0, reuse=False, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.name_scope(name_scope):
        with tf.name_scope("expand_input"):
            xt_in = tf.zeros([1, z_dim], dtype=tf.float32)
            #Expand dims for RNN
            if xt is not None:
                #Used to transform CNN representation
                P_W1 = tf.get_variable('P_W1', shape=[xt.get_shape().as_list()[1], z_dim], initializer=initializer)
                P_b1 = tf.get_variable('P_b1', shape=[z_dim], initializer=initializer)
                xt_ = tf.matmul(xt, P_W1) + P_b1
                #Make the initial state all 0, just use conditioning noise
                xt_in = tf.concat([xt_in, xt_], axis=0)
            xt_in = tf.expand_dims(xt_in, 0)

        with tf.name_scope("Decoder_RNN"):
            #Condition on the noise z
            z = tf.reshape(z, [1, z_dim])
            dec_init_state = rnn.ConditionalGRUState(h=tf.zeros([1, h_dim], dtype=tf.float32), c=z)
            dec_outs, _ = bidirectional_conditional_gru(xt_in, size=h_dim, name="gru_dec", reuse=reuse, initial_state=dec_init_state)
            print(dec_outs)
            dec_outs = tf.reshape(dec_outs, [-1, 2*h_dim])

            drop = tf.nn.dropout(dec_outs, keep_prob=keep_prob)

            P_W2 = tf.get_variable('P_W2', shape=[2*h_dim, X_dim], initializer=initializer)
            P_b2 = tf.get_variable('P_b2', shape=[X_dim], initializer=initializer)

        logits = tf.matmul(drop, P_W2) + P_b2
        prob = tf.nn.sigmoid(logits)
        return prob, logits

def cnn_vae_rnn(X, z, z_rnn_samples, X_dim, z_dim=512, h_dim=512, keep_prob=1.0, initializer=tf.contrib.layers.xavier_initializer()):
    '''
    Model definition for VRNN with regular GRU decoder
    '''
    with tf.variable_scope('vae_rnn') as scope:
        S = X.get_shape().as_list()
        X_in = tf.reshape(X, shape=[-1, S[-3], S[-2], S[-1]])        

        latent_feature_enc = midi_latent_feature(X_in, z_dim=z_dim, name='CNN_Encoder', keep_prob=keep_prob)
        
        ldim = latent_feature_enc.get_shape().as_list()[-1]

        latent_feature_enc = tf.reshape(latent_feature_enc, shape=[-1, S[1], ldim])
        
        z_mu, z_logvar = Q(latent_feature_enc, z_dim=z_dim, h_dim=h_dim, name_scope="Encoder_train", keep_prob=keep_prob)
        z_sample = sample_z(z_mu, z_logvar)
    
        out_samples, logits = P(z_sample, X_dim, z_dim=z_dim, h_dim=h_dim, name_scope="Decoder_train", xt=latent_feature_enc[:,:-1,:], keep_prob=keep_prob)
    
        # Sampling from random z
        scope.reuse_variables()
        
        S_zs = z_rnn_samples.get_shape().as_list()
        z_rnn_samples = tf.reshape(z_rnn_samples, shape=[-1, S_zs[-3], S_zs[-2], S_zs[-1]])
        latent_z_rnn = midi_latent_feature(z_rnn_samples, z_dim=z_dim, name='CNN_Encoder', reuse=True)
        zldim = latent_z_rnn.get_shape().as_list()[-1]
        latent_z_rnn = tf.reshape(latent_z_rnn, shape=[-1, S_zs[1], zldim])
        X_samples, _ = P(z, X_dim, name_scope="Decoder_test", xt=latent_feature_enc[:,:-1,:], reuse=True, z_dim=z_dim, h_dim=h_dim)

        return dict(X_samples=X_samples, out_samples=out_samples, 
            logits=logits, z_mu=z_mu, z_logvar=z_logvar)

def conditional_cnn_vae_rnn(X, z, z_rnn_samples, X_dim, z_dim=512, h_dim=512, keep_prob=1.0, initializer=tf.contrib.layers.xavier_initializer()):
    '''
    Model definition for VRNN with conditional GRU decoder
    '''
    with tf.variable_scope('vae_rnn') as scope:
        
        latent_feature_enc = midi_latent_feature(X, z_dim=z_dim, name='CNN_Encoder', keep_prob=1.0)
    
        z_mu, z_logvar = Q(latent_feature_enc, z_dim=z_dim, name_scope="Encoder_train", keep_prob=1.0)
        z_sample = sample_z(z_mu, z_logvar)
    
        out_samples, logits = P_conditional(z_sample, X_dim, z_dim=z_dim, h_dim=h_dim, name_scope="Decoder_train", xt=latent_feature_enc[:-1], keep_prob=keep_prob)
    
        # Sampling from random z
        scope.reuse_variables()
        latent_z_rnn = midi_latent_feature(z_rnn_samples, z_dim=z_dim, name='CNN_Encoder', reuse=True)
        X_samples, _ = P_conditional(z, X_dim, name_scope="Decoder_test", xt=latent_z_rnn, reuse=True, z_dim=z_dim, h_dim=h_dim)

        return dict(X_samples=X_samples, out_samples=out_samples, 
            logits=logits, z_mu=z_mu, z_logvar=z_logvar)
