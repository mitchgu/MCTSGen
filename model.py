from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=28, input_width=28, crop=True,
         batch_size=64, sample_num = 1, output_height=28, output_width=28,
         y_dim=10, z_dim=100, gf_dim=16, df_dim=16,
         gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='mnist',
         input_fname_pattern='*.jpg', checkpoint_dir='checkpoint', sample_dir='samples'):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    self.d_bn3 = batch_norm(name='d_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    self.data_X, self.data_y = self.load_mnist()
    self.c_dim = self.data_X[0].shape[-1]

    self.grayscale = (self.c_dim == 1)

    self.build_model()

    # success, counter = self.load(self.checkpoint_dir)
    # if not success:
    #   raise Exception("Unable to load pretrained discriminator from checkpoints")
    
    np.random.seed()

  def build_model(self):
    self.y = tf.placeholder(tf.float32, [self.batch_size, 1], name='labels')

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='images')

    inputs = self.inputs

    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    
    self.d_sum = histogram_summary("d", self.D)

    self.grad = tf.gradients(self.D, self.inputs)[0]

    def sigmoid_cross_entropy_with_logits(x, y):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

    self.d_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, self.y))

    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    self.t_vars = tf.trainable_variables()

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.t_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.writer = SummaryWriter("./logs", self.sess.graph)
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]

        # Update D network
        _, errD, summary_str = self.sess.run([d_optim, self.d_loss, self.d_sum],
          feed_dict={ 
            self.inputs: batch_images,
            self.y:batch_labels,
          })
        self.writer.add_summary(summary_str, counter)

        # errD = self.d_loss.eval({
        #     self.y:batch_labels
        # })

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f," \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD))

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()


      image = tf.reshape(image, [-1, 28, 28, 1])
      conv_1 = conv(image, 5, 5, 1, 32, 'conv1')
      pool_1 = pool2(conv_1, 'pool1')
      
      conv_2 = conv(pool_1, 5, 5, 32, 64, 'conv2')
      pool_2 = pool2(conv_2, 'pool2')
      
      fc1 = fc(flatten(pool_2, 7*7*64), 7*7*64, 1024, 'fc1', True)  
      # drop_1 = drop(fc1, keep_prob, 'drop1')

      fc2 = fc(fc1, 1024, 1, 'fc2', False) 
      logits = tf.identity(fc2)

      # h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      # h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      # h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      # h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      # h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(logits), logits

  def run_policy(self, samples, y):
    v, grad = self.sess.run([self.D, self.grad], feed_dict={
      self.inputs: samples
      })
    return v, grad

  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'th-train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    # fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd,dtype=np.uint8)
    # trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'th-t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    # fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd,dtype=np.uint8)
    # teY = loaded[8:].reshape((10000)).astype(np.float)

    # trY = np.asarray(trY)
    # teY = np.asarray(teY)
    
    fakeX = np.load('generation0.npy')

    X = np.concatenate((trX, teX, fakeX), axis=0)
    y = np.concatenate((np.ones((70000,1)), np.zeros((fakeX.size,1)))).astype(np.int)
    #np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    # y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    # for i, label in enumerate(y):
    #   y_vec[i,y[i]] = 1.0
    
    return X/255.,y

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
