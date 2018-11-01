#/usr/bin/python

'''
	This script contains the code to train and evaluate a variational autoencoder on
	the MNIST dataset

	Protypes:

	Todo:

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


class GAN:
    def __init__(self, z_dim=100, lr=0.001, epochs=75, g_units=[512,512,512], \
                d_units=[512,512,512], batch_size=16, n_input=784, dropout=0, \
                load=0, save=0, verbose=0):
        self.z_dim = z_dim
        self.lr = lr
        self.epochs = epochs
        self.g_units = g_units
        self.d_units = d_units
        self.batch_size = batch_size
        self.n_input = n_input
        self.load = load
        self.save = save
        self.dropout = dropout
        self.verbose = verbose

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    # USAGE:
    #       - generate random uniform vectors for input to generator
    # PARAMS:
    #   m: minibatch size
    #   n: vector dim
    def sample_z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])


    # USAGE:
    #       - generate data from random noise
    # PARAMS:
    #   z: input sample
    def generator(self, z, alpha=0.01):
        with tf.variable_scope('generator'):
            layer = z
            for idx, n in enumerate(self.g_units):
                layer = tf.layers.dense(inputs=layer, units=n, activation=tf.nn.relu, name='G_' + str(idx))
                #layer = tf.maximum(layer, layer * alpha)

            g_prob = tf.layers.dense(inputs=layer, units=self.n_input, activation=tf.nn.sigmoid, name='G_' + str(idx + 1))

            return g_prob


    # USAGE:
    #       - discriminate between real and fake
    # PARAMS:
    #   x: input data sample from X or G(z)
    def discriminator(self, x, alpha=0.01, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            layer = x
            for idx, n in enumerate(self.d_units):
                layer = tf.layers.dense(inputs=layer, units=n, activation=tf.nn.relu, name='D_' + str(idx))
                # layer = tf.maximum(layer, layer * alpha)
                if self.dropout:
                    layer = tf.nn.dropout(layer, self.dropout)

            # use a sigmoid activation to get the data between 0 and 1
            d_logit = tf.layers.dense(inputs=layer, units=1, activation=None, name='D_' + str(idx + 1))
            d_prob = tf.nn.sigmoid(d_logit)

        return d_logit, d_prob


    def train(self, dataset):
        # define placeholders for input data
        x = tf.placeholder("float", [None, self.n_input])
        z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        # create graph for generated samples and discriminator for real and fake samples
        g_sample = self.generator(z)
        d_real_logit, d_real_prob = self.discriminator(x) 
        d_fake_logit, d_fake_prob = self.discriminator(g_sample, reuse=True) 

        # label smoothing trick... use 0.9 instead of 1
        smooth = 0.1
        d_labels_real = tf.ones_like(d_real_logit) * (1 - smooth)

        # discriminator loss log(D(x)) + log(D(1 - G(z)))
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logit, labels=d_labels_real))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logit, labels=tf.zeros_like(d_fake_logit)))
        d_loss = d_loss_fake + d_loss_real

        # generator loss D(G(z))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logit, labels=tf.ones_like(d_fake_logit)))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'D_' in var.name]
        g_vars = [var for var in t_vars if 'G_' in var.name]

        d_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
        g_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('out/'):
            os.makedirs('out/')

        i = 0

        for it in range(100000):
            batch_x, _ = dataset.train.next_batch(self.batch_size)
            _, dl = sess.run([d_opt, d_loss], feed_dict={x: batch_x, z: self.sample_z(self.batch_size, self.z_dim)})
            _, gl = sess.run([g_opt, g_loss], feed_dict={z: self.sample_z(self.batch_size, self.z_dim)})

            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('d_loss: {:.4}'. format(dl))
                print('g_loss: {:.4}'.format(gl))
                print('----------')

                samples = sess.run(g_sample, feed_dict={z: self.sample_z(16, self.z_dim)})

                fig = plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                i += 1


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
gan = GAN(g_units=[256], d_units=[256])
#gan.train(mnist)










