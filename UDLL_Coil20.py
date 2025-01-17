# Code Authors: Pan Ji,     University of Adelaide,         pan.ji@adelaide.edu.au
#               Tong Zhang, Australian National University, tong.zhang@anu.edu.au
# Copyright Reserved!
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from sklearn import cluster
from munkres import Munkres
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from tensorflow.examples.tutorials.mnist import input_data
import  os

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

class ConvAE(object):
	def __init__(self, n_input, kernel_size, n_hidden, reg_const1 = 1.0, reg_const2 = 1.0,reg3 = 10, reg = None, batch_size = 256, L = None,\
		denoise = False, model_path = None, logs_path = None):	
	#n_hidden is a arrary contains the number of neurals on every layer
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.reg = reg
		self.model_path = model_path		
		self.kernel_size = kernel_size		
		self.iter = 0
		self.batch_size = batch_size
		weights = self._initialize_weights()
		
		# model
		self.x = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
		self.learning_rate = tf.placeholder(tf.float32, [])
		
		if denoise == False:
			x_input = self.x
			latent, shape = self.encoder(x_input, weights)

		else:
			x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
											   mean = 0,
											   stddev = 0.2,
											   dtype=tf.float32))

			latent,shape = self.encoder(x_input, weights)
		self.z_conv = tf.reshape(latent,[batch_size, -1])		
		self.z_ssc, Coef = self.selfexpressive_moduel(batch_size)
		Coef = tf.divide(tf.add(Coef, tf.transpose(Coef)), 2.0)
		Coef = tf.subtract(Coef, tf.diag(tf.diag_part(Coef)))
#		D = tf.diag(tf.reduce_sum(Coef, 0))
#		invD = tf.sqrt(tf.matrix_inverse(D))
#		Coef = tf.matmul(invD, tf.matmul(Coef, invD))
		
		self.Coef = Coef						
		latent_de_ft = tf.reshape(self.z_ssc, tf.shape(latent))		
		self.x_r_ft = self.decoder(latent_de_ft, weights, shape)		
				

		self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]) 
			
#		self.z_conv_ = tf.transpose(tf.divide(tf.transpose(self.z_conv), D))
		
		self.recon_ssc =  tf.reduce_sum(tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))
		self.sc_loss = tf.trace(tf.matmul(tf.transpose(self.z_conv), tf.matmul(L, self.z_conv)))
		self.recon_sc = tf.reduce_sum(tf.pow(tf.subtract(self.z_conv, self.z_ssc), 2.0))
		self.reg_ssc = tf.reduce_sum(tf.pow(self.Coef,2))
		tf.summary.scalar("reg_lose", self.reg_ssc)		
		
		self.loss_ssc = self.sc_loss*reg_const2 * 2 + reg_const1*self.reg_ssc + self.recon_ssc + self.recon_sc * reg3

		self.merged_summary_op = tf.summary.merge_all()		
		self.optimizer_ssc = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_ssc)
		self.init = tf.global_variables_initializer()
		self.sess = tf.InteractiveSession()
		self.sess.run(self.init)
		self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	def _initialize_weights(self):
		all_weights = dict()
		all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, n_hidden[0]],
			initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
		all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))
		

		all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[0], self.kernel_size[0],1, n_hidden[0]],
			initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
		all_weights['dec_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))
		return all_weights


	# Building the encoder
	def encoder(self,x, weights):
		shapes = []
		# Encoder Hidden layer with relu activation #1
		shapes.append(x.get_shape().as_list())
		layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
		layer1 = tf.nn.relu(layer1)
		return  layer1, shapes

	# Building the decoder
	def decoder(self,z, weights, shapes):
		# Encoder Hidden layer with relu activation #1
		shape_de1 = shapes[0]
		layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(self.x)[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
		 strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
		layer1 = tf.nn.relu(layer1)
		
		return layer1



	def selfexpressive_moduel(self,batch_size):
		
		Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')			
		z_ssc = tf.matmul(Coef,	self.z_conv)
		return z_ssc, Coef


	def finetune_fit(self, X, lr):
		C,l1_cost, l2_cost, summary, _ = self.sess.run((self.Coef, self.reg_ssc, self.sc_loss, self.merged_summary_op, self.optimizer_ssc), \
													feed_dict = {self.x: X, self.learning_rate: lr})
		self.summary_writer.add_summary(summary, self.iter)
		self.iter = self.iter + 1
		return C, l1_cost,l2_cost 
		
	def get_embeding(self, X):
		xembeding = self.sess.run(self.z_conv, feed_dict = {self.x:X})
		return xembeding
	
	def initlization(self):
		tf.reset_default_graph()
		self.sess.run(self.init)	

	def transform(self, X):
		return self.sess.run(self.z_conv, feed_dict = {self.x:X})

	def save_model(self):
		save_path = self.saver.save(self.sess,self.model_path)
		print ("model saved in file: %s" % save_path)

	def restore(self):
		self.saver.restore(self.sess, self.model_path)
		print ("model restored")
		

def best_map(L1,L2):
	#L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2

def thrC(C,ro):
	if ro < 1:
		N = C.shape[1]
		Cp = np.zeros((N,N))
		S = np.abs(np.sort(-np.abs(C),axis=0))
		Ind = np.argsort(-np.abs(C),axis=0)
		for i in range(N):
			cL1 = np.sum(S[:,i]).astype(float)
			stop = False
			csum = 0
			t = 0
			while(stop == False):
				csum = csum + S[t,i]
				if csum > ro*cL1:
					stop = True
					Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
				t = t + 1
	else:
		Cp = C

	return Cp

def post_proC(C, K, d, alpha):
	# C: coefficient matrix, K: number of clusters, d: dimension of each subspace
	n = C.shape[0]
	C = 0.5*(C + C.T)	 
	C = C - np.diag(np.diag(C)) + np.eye(n,n) # for sparse C, this step will make the algorithm more numerically stable
	r = d*K + 1		
	U, S, _ = svds(C,r,v0 = np.ones(n))
	U = U[:,::-1] 
	S = np.sqrt(S[::-1])
	S = np.diag(S)
	U = U.dot(S)
	U = normalize(U, norm='l2', axis = 1)  
	Z = U.dot(U.T)
	Z = Z * (Z>0)
	L = np.abs(Z ** alpha)
	L = L/L.max()
	L = 0.5 * (L + L.T)	
	spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
	spectral.fit(L)
	grp = spectral.fit_predict(L) + 1
	return grp, L

def err_rate(gt_s, s):
	c_x = best_map(gt_s,s)
	err_x = np.sum(gt_s[:] != c_x[:])
	missrate = err_x.astype(float) / (gt_s.shape[0])
	return missrate  

data = sio.loadmat('COIL20.mat')
graph = sio.loadmat('COIL20_graph.mat')
Img = data['fea']
Label = data['gnd']
Img = np.reshape(Img,(Img.shape[0],32,32,1))
#D = np.diag(np.sum(graph['A'],1))
L = np.float32(graph['L'])


n_input = [32,32]
kernel_size = [3]
n_hidden = [15]
batch_size = 20*72
model_path = './pretrain-model-COIL20/model.ckpt'
ft_path = './pretrain-model-COIL20/model.ckpt'
logs_path = './pretrain-model-COIL20/logs'

num_class = 20 #how many class we sample
num_sa = 72

batch_size_test = num_sa * num_class


iter_ft = 0
ft_times = 68
display_step = 34
alpha = 0.04

learning_rate = 1e-3
reg1 = 1.0
reg2 = 9.0
reg3 = 1000

CAE = ConvAE(n_input = n_input, n_hidden = n_hidden, reg_const1 = reg1, reg_const2 = reg2,reg3 = reg3, kernel_size = kernel_size, \
			batch_size = batch_size_test, L = L, model_path = model_path, logs_path= logs_path)

acc_= []
for i in range(0,1):
	coil20_all_subjs = Img
	coil20_all_subjs = coil20_all_subjs.astype(float)	
	label_all_subjs = Label
	label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
	label_all_subjs = np.squeeze(label_all_subjs) 
	 	
	CAE.initlization()
	CAE.restore()
	
	for iter_ft  in range(ft_times):
		iter_ft = iter_ft+1
		C,l1_cost,l2_cost = CAE.finetune_fit(coil20_all_subjs,learning_rate)
		if iter_ft % display_step == 0:
			print ("epoch: %.1d" % iter_ft, "cost: %.8f" % (l1_cost/float(batch_size_test)))
			C = thrC(C,alpha)
			
			y_x, CKSym_x = post_proC(C, num_class, 12 ,8)
			missrate_x = err_rate(label_all_subjs,y_x)
						
			ac = 1 - missrate_x
			print ("experiment: %d" % i,"acc: %.4f" % ac)
			acc_.append(ac)

embed = CAE.transform(coil20_all_subjs)
acc_ = np.array(acc_)
m = np.mean(acc_)
me = np.median(acc_)
print(m)
print(me)
print(acc_.max())

sio.savemat('coil20em1',{'X':embed,'Y':label_all_subjs})
#sio.savemat('2W',{'W1':C, 'W2':CKSym_x})
#sio.savemat('W.mat',{'W':C})
#eng = matlab.engine.start_matlab()
#C1 = matlab.double(C.tolist())
#labels = matlab.double(label_all_subjs.tolist())
#acc1 = eng.CLR(C1, 20, labels)
#print(acc1)

		
	
