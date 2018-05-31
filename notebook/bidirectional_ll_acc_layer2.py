
# coding: utf-8

# In[1]:

import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from model.vae import cnn_vae_rnn
from util.miditools import piano_roll_to_pretty_midi


# In[2]:

##################################################################
# Setting up constants and loading the data
##################################################################

# nintendo
nintendo_file = './Downloads/node-vgmusic-downloader/download/console/nintendo/gameboy/'
# nottingham
nottingham_file = './Downloads/node-vgmusic-downloader/download/console/microsoft/xbox/'

# 'wget '+'http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle -O '
# 'wget '+'http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle -O '
# 'wget '+'http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle -O '
# 'wget '+'http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle -O '


# In[3]:

snapshot_interval = 200
log_interval = 50

checkpoint_file = './tfmodel/exp-new-bigru-iter-2000-0520.tfmodel'
# mudb_file = '../Nottingham/preprocessing/CN_mudb_train.npz'
dev_file = '../Nottingham/preprocessing/CN_mudb_valid.npz' # success!!!
# dev_file = '/home/eko/Downloads/mudb_train.npz'

# train_data = np.load(mudb_file)
dev_data = np.load(dev_file)
# print range(train_data['bars'])


# In[4]:

# fs = train_data['fs']
fs = dev_data['fs']
print fs

num_timesteps = int(fs)
# bars = train_data['bars']
devBars = dev_data['bars']
# np.random.shuffle(bars)

print devBars.shape
# print len(bars)


# In[5]:

note_range = int(devBars.shape[2])


# T = int(train_data['T']) #16
T = int(dev_data['T']) #16

# num_batches = int(bars.shape[0])
num_batches = int(devBars.shape[0])

height = num_timesteps #19
width = note_range #128
n_visible = note_range * num_timesteps
n_epochs = 100

z_dim = 350
X_dim = width * height
n_hidden = z_dim
h_dim = z_dim
batch_size = 32

trainBarsBatch = np.reshape(devBars, (-1, T, height, width, 1))
trainBarsBatches = []
i = 0
while i < trainBarsBatch.shape[0] - 32:
    trainBarsBatches.append(trainBarsBatch[i:i+32])
    i += 32
devBarsBatch = np.reshape(devBars, (-1, T, height, width, 1))
devBarsBatches = []
i = 0
while i < devBarsBatch.shape[0] - 32:
    devBarsBatches.append(devBarsBatch[i:i+32])
    i += 32
#devBarsBatch = np.array_split(devBarsBatch, batch_size)
initializer = tf.contrib.layers.xavier_initializer()

audio_sr = 44100

devLoss = True
devInterval = 100


# In[6]:

##################################################################
# Loading the model
##################################################################
with tf.name_scope('placeholders'):
    z = tf.placeholder(tf.float32, shape=[None, z_dim], name="Generated_noise")
    #(batch x T x width x height x channels)
    z_rnn_samples = tf.placeholder(tf.float32, shape=[None, T, height, width, 1], name="Generated_midi_input")
    
    X = tf.placeholder(tf.float32, shape=[None, T, height, width, 1], name="Training_samples")
    kl_annealing = tf.placeholder(tf.float32, name="KL_annealing_multiplier")

    
# model selection
model = cnn_vae_rnn(X, z, z_rnn_samples, X_dim, z_dim=z_dim, h_dim=h_dim, initializer=initializer, keep_prob=1.0)
# model = cnn_vae_rnn(X, z, z_rnn_samples, X_dim, z_dim=z_dim, h_dim=h_dim, initializer=initializer, keep_prob=1.0)
# model = cnn_vae
# model = cnn_rnn
# model = vae_rnn
# model = vae
# model = rnn


X_samples, out_samples, logits = (model['X_samples'], model['out_samples'], model['logits'])
z_mu, z_logvar = (model['z_mu'], model['z_logvar'])


# In[7]:

##################################################################
# Losses
##################################################################
with tf.name_scope("Loss"):
    X_labels = tf.reshape(X, [-1, width*height])

    with tf.name_scope("cross_entropy"):
        recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X_labels), 1)
    with tf.name_scope("kl_divergence"):
        kl_loss = kl_annealing * 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.exp(z_logvar) - z_logvar - 1.,1) 
    
    true_note = tf.argmax(X_labels,1)
    pred_note = tf.argmax(out_samples,1)
    correct_pred = tf.equal(pred_note, true_note)
    
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # accuracy
    
    
    recon_loss = tf.reduce_mean(tf.reshape(recon_loss, [-1, T]), axis=1)
    loss = tf.reduce_mean(recon_loss + kl_loss)


# print accuracy    


# In[8]:

##################################################################
# Optimizer
##################################################################
with tf.name_scope("Optimizer"):
    solver = tf.train.AdamOptimizer()
    grads = solver.compute_gradients(loss)
    grads = [(tf.clip_by_norm(g, clip_norm=1), v) for g, v in grads]
    train_op = solver.apply_gradients(grads)

##################################################################
# Logging
##################################################################
with tf.name_scope("Logging"):
    recon_loss_ph = tf.placeholder(tf.float32)
    kl_loss_ph = tf.placeholder(tf.float32)
    loss_ph = tf.placeholder(tf.float32)
    audio_ph = tf.placeholder(tf.float32)
#     acc_ph = tf.placeholder(tf.float32)

    tf.summary.scalar("Reconstruction_loss", recon_loss_ph)
    tf.summary.scalar("KL_loss", kl_loss_ph)
    tf.summary.scalar("Loss", loss_ph)
#     tf.summary.scalar("Acc", acc_ph)
    
    tf.summary.audio("sample_output", audio_ph, audio_sr)
    log_op = tf.summary.merge_all()

writer = tf.summary.FileWriter('./tb/', graph=tf.get_default_graph())

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

# Run Initialization operations
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

loss_avg = 0.0
decay = 0.99
min_loss = 100.0
min_dev_loss = 200.0
time0 = time.time()
##################################################################
# Optimization loop
##################################################################
i = 0
accuracy = 0
for e in range(n_epochs):
    print("%s EPOCH %d %s" % ("".join(10*["="]), e, "".join(10*["="])))
    for batch in trainBarsBatches:
        kl_an = 1.0#min(1.0, (i / 10) / 200.)
        _,loss_out, kl, recon, acc_out = sess.run([train_op, loss, kl_loss, recon_loss, accuracy_op], feed_dict={X: batch, kl_annealing: kl_an})
        
        if (i % log_interval) == 0:
            loss_avg = decay*loss_avg + (1-decay)*loss_out

            
            print('\titer = %d, accuracy = %f' % (i, acc_out))
            
#             print('\titer = %d, perplexity = %f' % (i, perplexity_out))
            
            print('\titer = %d, local_loss (cur) = %f, local_loss (avg) = %f, kl = %f'
                % (i, loss_out, loss_avg, np.mean(kl)))
            
            time_spent = time.time() - time0
            print('\n\tTotal time elapsed: %f sec. Average time per batch: %f sec\n' %
                (time_spent, time_spent / (i+1)))
                    
 
            #Random samples
            z_in = np.random.randn(1, z_dim)
            z_rnn_out = np.zeros((T,height,width,1))
            first = True
            for j in range(T):
                z_rnn_out = np.expand_dims(z_rnn_out, axis=0)
                samples = sess.run(X_samples, feed_dict={z: np.random.randn(1, z_dim), X: z_rnn_out})
                
                
                frames = j + 1
                samples = samples.reshape((-1, height, width, 1))
                z_rnn_out = np.concatenate([samples[:frames], np.zeros((T-frames, height, width, 1))])

            samples = samples.reshape((num_timesteps*(T), note_range))
            thresh_S = samples >= 0.5
            
            pm_out = piano_roll_to_pretty_midi(thresh_S.T * 127, fs=fs)
            midi_out = './tb/audio/test002_{0}.mid'.format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S"))
            wav_out = './tb/audio/test002_{0}.wav'.format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S"))
            audio = pm_out.synthesize() 
            audio = audio.reshape((1, len(audio)))
            #Write out logs
#             summary = sess.run(log_op, feed_dict={recon_loss_ph: np.mean(recon), kl_loss_ph: np.mean(kl),
#                                                  loss_ph: loss_out, audio_ph: audio})
#             writer.add_summary(summary, i)
        
        if devLoss and i % devInterval == 0:
            #dls = []
            #for dbatch in devBarsBatches:
            #    dev_loss_out, kl, recon = sess.run([loss, kl_loss, recon_loss], feed_dict={X: dbatch, kl_annealing: kl_an})
            #    dls.append(dev_loss_out)
            #dev_loss_out = sum(dls) / len(dls)
            #print("Dev set loss %.2f" % dev_loss_out)

            if loss_out < min_dev_loss:
                print("Saving checkpoint with train loss %d" % loss_out)
                min_dev_loss = loss_out
                
        i += 1
        saver.save(sess, checkpoint_file)


# In[ ]:

with tf.Session() as sess:
    saver.restore(sess, checkpoint_file)

#     saver = tf.train.import_meta_graph('./tfmodel/exp-new-bigru-iter-2000-0412.tfmodel.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('./tfmodel/'))

    
    #Generate T frames

    T=16
    #Random samples
    z_in = np.random.randn(1, z_dim)
    z_rnn_out = np.zeros((T,height,width,1))
    first = True
    
    for j in range(T):
        z_rnn_out = np.expand_dims(z_rnn_out, axis=0)
        samples = sess.run(X_samples, feed_dict={z: np.random.randn(1, z_dim), X: z_rnn_out})
        frames = j + 1
        samples = samples.reshape((-1, height, width, 1))
        z_rnn_out = np.concatenate([samples[:frames], np.zeros((T-frames, height, width, 1))])
        
        
        
    samples = samples.reshape((num_timesteps*(T), note_range))
    thresh_S = samples >= 0.5
    
    print np.shape(thresh_S)
    
#     plt.figure(figsize=(36,6))
#     plt.subplot(1,2,1)
#     plt.imshow(sams)
#     plt.subplot(1,2,2)
#     plt.imshow(thresh_S)
#     plt.tight_layout()
#     plt.pause(0.1)
    pm = piano_roll_to_pretty_midi(thresh_S.T, fs)
    
    
    for i in range (10):
        pm.write('./output/test04_0414_{0}.mid'.format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S")))


# In[ ]:

# sequence generation

# generate step for batch
# event_sequences
# inputs

# final_state
# log-likelihood: the log-likelihood of the chosen softmax value for each event sequence, 1-D array 
# of length.


# In[ ]:

def evaluate_log_likelihood(self, event_sequences, softmax):
    """Evaluate the log likelihood of multiple event sequences.
    Each event sequence is evaluated from the end. If the size of the
    corresponding softmax vector is 1 less than the number of events, the entire
    event sequence will be evaluated (other than the first event, whose
    distribution is not modeled). If the softmax vector is shorter than this,
    only the events at the end of the sequence will be evaluated.
    Args:
      event_sequences: A list of EventSequence objects.
      softmax: A list of softmax probability vectors. The list of softmaxes
          should be the same length as the list of event sequences.
    Returns:
      A Python list containing the log likelihood of each event sequence.
    Raises:
      ValueError: If one of the event sequences is too long with respect to the
          corresponding softmax vectors.
    """
    all_loglik = []
    for i in range(len(event_sequences)):
      if len(softmax[i]) >= len(event_sequences[i]):
        raise ValueError(
            'event sequence must be longer than softmax vector (%d events but '
            'softmax vector has length %d)' % (len(event_sequences[i]),
                                               len(softmax[i])))
      end_pos = len(event_sequences[i])
      start_pos = end_pos - len(softmax[i])
      loglik = 0.0
      for softmax_pos, position in enumerate(range(start_pos, end_pos)):
        index = self.events_to_label(event_sequences[i], position)
        loglik += np.log(softmax[i][softmax_pos][index])
      all_loglik.append(loglik)
    return all_loglik


# In[ ]:

# each event sequence is evaluated from the end
# if the size of the corresponding softmax vector is 1 less than the number of events,
# the entire event sequence will be evaluated

# event_sequences: eventsequence 
# softmax: softmax probability vectors
# The list of softmaxes


# In[13]:

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file('./tfmodel/exp-new-bigru-iter-2000-0412.tfmodel', tensor_name='', all_tensors=True)

