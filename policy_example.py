import os
import numpy as np
import tensorflow as tf
from model import DCGAN

NPY_DIR = 'npy'

with tf.Session() as sess:
  # Create DCGAN model
  dcgan = DCGAN(sess)

  for samples_fname in ["samples1", "samples2", "samples3"]:
    # Load samples from numpy file
    samples = np.load(os.path.join(NPY_DIR, samples_fname + ".npy"))
    samples_y = np.load(os.path.join(NPY_DIR, samples_fname + "_y.npy"))

    # Samples needs to be shape (64, 28, 28, 1)
    # labels need to be in shape (64)
    v, grad = dcgan.run_policy(samples, samples_y) # returns real/fake value and gradients

    # Print mean value for 64 samples
    print("Mean real/fake value for {}: {}".format(samples_fname, np.mean(v)))
    # Dump the gradients to a file (64, 28, 28, 1)
    grad.dump(os.path.join(NPY_DIR, samples_fname + "_grad.npy"))
