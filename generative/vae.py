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


class VAE:
	def __init__(self, latent_dim=2, lr=0.001, epochs=75, h_units=[512,512,512], \
				batch_size=16, n_input=784, dropout=0, load=0, save=0, verbose=0):
		self.latent_dim = latent_dim
		self.lr = lr
		self.epochs = epochs
		self.h_units = h_units
		self.batch_size = batch_size
		self.n_input = n_input
		self.load = load
		self.save = save
		self.dropout = dropout
		self.verbose = verbose

		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def inference_network(self, x):
		with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
			layer = x
			for n in self.h_units:
				layer = tf.layers.dense(inputs=layer, units=n, activation=tf.nn.relu)

			gaussian_params = tf.layers.dense(inputs=layer, units=self.latent_dim * 2, activation=None)

			mu = gaussian_params[:, :self.latent_dim]

			# std dev must be positive... parametrize with a softplus
			sigma = tf.nn.softplus(gaussian_params[:, self.latent_dim:])

			return mu, sigma


	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def inference_convnet(self, x):
		with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
			input_layer = tf.reshape(x, [-1, 28, 28, 1])

			conv1 = tf.layers.conv2d(
								inputs=input_layer,
								filters=32,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)

			pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
			
			conv2 = tf.layers.conv2d(
								inputs=pool1,
								filters=64,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)

			pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

			pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

			gaussian_params = tf.layers.dense(inputs=pool2_flat, units=self.latent_dim * 2, activation=None)

			mu = gaussian_params[:, :self.latent_dim]

			# std dev must be positive... parametrize with a softplus
			sigma = tf.nn.softplus(gaussian_params[:, self.latent_dim:])

			return mu, sigma


	# USAGE:
	# 		- decoder network for vae
	# PARAMS:
	#	z: input latent variable
	#	n_hidden: LIST of num. neurons per hidden layer
	def generative_network(self, z):
		with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
			layer = z
			units = self.h_units[:]
			units.reverse()
			for n in units:
				layer = tf.layers.dense(inputs=layer, units=n, activation=tf.nn.relu)

			# use a sigmoid activation to get the data between 0 and 1
			logits = tf.layers.dense(inputs=layer, units=self.n_input, activation=None)
			probs = tf.nn.sigmoid(logits)

		return probs, logits


	# USAGE:
	# 		- decoder network for vae on MNIST data
	# PARAMS:
	#	z: input latent variable
	#	n_hidden: LIST of num. neurons per hidden layer
	def generative_convnet(self, z):
		with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
			layer = tf.layers.dense(inputs=z, units=7*7*32, activation=tf.nn.relu)

			layer_reshape = tf.reshape(layer, [-1, 7, 7, 32])

			layer1 = tf.layers.conv2d_transpose(layer_reshape, kernel_size=3, filters=64, \
												strides=2, padding="same", activation=tf.nn.relu)

			layer2 = tf.layers.conv2d_transpose(layer1, kernel_size=3, filters=64, \
												strides=2, padding="same", activation=tf.nn.relu)

			layer3 = tf.layers.conv2d_transpose(layer2, kernel_size=3, filters=1, \
												strides=1, padding="same")

			logits = tf.reshape(layer3, [-1, 28 * 28])

			probs = tf.nn.sigmoid(logits)

		return probs, logits


	# USAGE:
	# 		- perform the reparameterization trick for vae
	# PARAMS:
	#	mean: mean produced by inference network
	#	sigma: sigma produced by inference network
	def reparameterize(self, mean, logvar):
		eps = tf.random_normal(shape=tf.shape(mean))
		q_z = mean + tf.exp(logvar * 0.5) * eps	
		return q_z


	def train(self, dataset):
		# define placeholders for input data
		x = tf.placeholder("float", [None, self.n_input])
		z = tf.placeholder(tf.float32, shape=[None, self.latent_dim])

		q_mu, q_sigma = self.inference_network(x=x)

		q_z = self.reparameterize(q_mu, q_sigma)

		_, x_logit = self.generative_network(q_z)

		X_samples, _ = self.generative_network(z)

		# define losses
		recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x), axis=1)
		kl = 0.5 * tf.reduce_sum(tf.exp(q_sigma) + tf.square(q_mu) - 1. - q_sigma, axis=1)

		ELBO = tf.reduce_mean(recon_loss + kl)

		loss = ELBO

		# optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

		saver = tf.train.Saver()

		sample_z = np.random.randn(16, self.latent_dim)

		# Initializing the variables
		init = tf.global_variables_initializer()

		sess = tf.Session()
		sess.run(init)

		for epoch in range(1, self.epochs + 1):
			avg_cost = 0.
			total_batch = int(dataset.train.num_examples/self.batch_size)

			for i in range(total_batch):
				batch_x, _ = dataset.train.next_batch(self.batch_size)
				_, c, rcl, kll = sess.run([optimizer, loss, recon_loss, kl], feed_dict={x: batch_x})

				avg_cost += c / total_batch

			print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))

			samples = sess.run(X_samples, feed_dict={z: sample_z})
			fig = plot(samples)
			plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
			plt.close(fig)

		saver.save(sess, "/tmp/model.ckpt")
		sess.close() 


	# run inference on the inference network Q(z | X)
	def infer(self, data):
		x = tf.placeholder("float", [None, self.n_input])
		q_mu, q_sigma = self.inference_network(x=x)

		saver = tf.train.Saver()

		sess = tf.Session()
		saver.restore(sess, "/tmp/model.ckpt")

		mu, sig = sess.run([q_mu, q_sigma], feed_dict={x: data})

		sess.close()

		return mu, sig


	# run inference on the generative network P(X | z)
	def generate(self):
		z = tf.placeholder(tf.float32, shape=[None, self.latent_dim])

		X_samples, _ = self.generative_network(z)

		saver = tf.train.Saver()

		sess = tf.Session()

		saver.restore(sess, "/tmp/model.ckpt")

		#mu, sig = 
		xsamp = sess.run(X_samples, feed_dict={z: np.random.randn(16, self.latent_dim)})

		fig = plot(xsamp)
		plt.savefig('out/{}.png'.format(str(0).zfill(3)), bbox_inches='tight')
		plt.close(fig)

		sess.close()



mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
vae = VAE(epochs=30, batch_size=64, h_units=[512, 256, 64], n_input=784, latent_dim=20)
vae.train(mnist)








