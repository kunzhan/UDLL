# Code Authors: Pan Ji,     University of Adelaide,         pan.ji@adelaide.edu.au
#               Tong Zhang, Australian National University, tong.zhang@anu.edu.au
# Copyright Reserved!
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
#from numpy.linalg import svd
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
#import matlab.engine

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1 = 1.0, re_constant2 = 1.0, reg3 = 1.0, L = None, batch_size = 200, reg = None, \
                denoise = False, model_path = None, restore_path = None, \
                logs_path = None):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        
        #input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        
        weights = self._initialize_weights()
        
        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                               mean = 0,
                                               stddev = 0.2,
                                               dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)
        
        #self.Coef = tf.Variable(np.eye(batch_size,batch_size,0,np.float32))    
        z = tf.reshape(latent, [batch_size, -1])  
        Coef = weights['Coef']         
        z_c = tf.matmul(Coef,z)    
        self.Coef = Coef       
        latent_c = tf.reshape(z_c, tf.shape(latent)) 
        self.z = z       
        
        self.x_r = self.decoder(latent_c, weights, shape)                
        
        # l_2 reconstruction loss 
        self.reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))
        tf.summary.scalar("recons_loss", self.reconst_cost)
       
        self.reg_losses = tf.reduce_sum(tf.pow(self.Coef,2.0))
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses )
        
        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0))
        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses )
		
        self.sc_loss = tf.trace(tf.matmul(tf.transpose(self.z), tf.matmul(L, self.z)))
        
        self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses  + self.sc_loss * reg3
        
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss) #GradientDescentOptimizer #AdamOptimizer
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)        
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]) 
        #[v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]       
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        all_weights['Coef']   = tf.Variable(0 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')        
        
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32)) # , name = 'enc_b0'
        
        iter_i = 1
        while iter_i < n_layers:
            enc_name_wi = 'enc_w' + str(iter_i)
            all_weights[enc_name_wi] = tf.get_variable(enc_name_wi, shape=[self.kernel_size[iter_i], self.kernel_size[iter_i], self.n_hidden[iter_i-1], \
                        self.n_hidden[iter_i]], initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
            enc_name_bi = 'enc_b' + str(iter_i)
            all_weights[enc_name_bi] = tf.Variable(tf.zeros([self.n_hidden[iter_i]], dtype = tf.float32)) # , name = enc_name_bi
            iter_i = iter_i + 1
        
        iter_i = 1
        while iter_i < n_layers:    
            dec_name_wi = 'dec_w' + str(iter_i - 1)
            all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers-iter_i], self.kernel_size[n_layers-iter_i], 
                        self.n_hidden[n_layers-iter_i-1],self.n_hidden[n_layers-iter_i]], initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
            dec_name_bi = 'dec_b' + str(iter_i - 1)
            all_weights[dec_name_bi] = tf.Variable(tf.zeros([self.n_hidden[n_layers-iter_i-1]], dtype = tf.float32)) # , name = dec_name_bi
            iter_i = iter_i + 1
            
        dec_name_wi = 'dec_w' + str(iter_i - 1)
        all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        dec_name_bi = 'dec_b' + str(iter_i - 1)
        all_weights[dec_name_bi] = tf.Variable(tf.zeros([1], dtype = tf.float32)) # , name = dec_name_bi
        
        return all_weights
        
    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        shapes.append(x.get_shape().as_list())
        layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layeri = tf.nn.relu(layeri)
        shapes.append(layeri.get_shape().as_list())
        
        n_layers = len(self.n_hidden)
        iter_i = 1
        while iter_i < n_layers:
            layeri = tf.nn.bias_add(tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[1,2,2,1],padding='SAME'),weights['enc_b' + str(iter_i)])
            layeri = tf.nn.relu(layeri)
            shapes.append(layeri.get_shape().as_list())
            iter_i = iter_i + 1
        
        layer3 = layeri
        return  layer3, shapes
    
    # Building the decoder
    def decoder(self,z, weights, shapes):
        n_layers = len(self.n_hidden)        
        layer3 = z
        iter_i = 0
        while iter_i < n_layers:
            #if iter_i == n_layers-1:
            #    strides_i = [1,2,2,1]
            #else:
            #    strides_i = [1,1,1,1]
            shape_de = shapes[n_layers - iter_i - 1]            
            layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack([tf.shape(self.x)[0],shape_de[1],shape_de[2],shape_de[3]]),\
                     strides=[1,2,2,1],padding='SAME'), weights['dec_b' + str(iter_i)])
            layer3 = tf.nn.relu(layer3)
            iter_i = iter_i + 1
        return layer3
    
    def partial_fit(self, X, lr): #  
        cost, summary, _, Coef = self.sess.run((self.reconst_cost, self.merged_summary_op, self.optimizer, self.Coef), feed_dict = {self.x: X, self.learning_rate: lr})#
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, Coef
		
    def get_embeding(self, X):
    		return self.sess.run(self.z, feed_dict = {self.x:X})
    
    
    def initlization(self):
        self.sess.run(self.init)
    
    def reconstruct(self,X):
        return self.sess.run(self.x_r, feed_dict = {self.x:X})
    
    def transform(self, X):
        return self.sess.run(self.z, feed_dict = {self.x:X})
    
    def save_model(self):
        save_path = self.saver.save(self.sess,self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")
        
def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
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
    C = 0.5*(C + C.T)
    r = min(d*K + 1, C.shape[0]-1)      
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
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
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

# def L2_distance(Z):
#     num = np.size(Z,0)
#     AA = np.sum(Z**2,axis=1)
#     AB = np.matmul(Z,Z.T)
#     A = np.tile(AA,(num,1))
#     B = A.T
#     d = A + B -2*AB
#     return d

# def constructW_PKN(Z,k):
#     num = np.size(Z,0)
#     print(num)

#     D = L2_distance(Z)
#     # dumb = np.sort(D)
#     idx = np.argsort(D)

#     W = np.zeros_like(D)

#     for i in range(num):
#         indx = idx[i,1:k+2]
#         di = D[i,indx]
#         W[i,indx] = (di[k]-di)/(k*di[k]-np.sum(di[0:k])+2.2204e-16);

#     W = (W + W.T)/2
#     return W
        
def test_face(Img, Label, CAE, num_class):       
    
    alpha = 0.2
    print(alpha)
    
    acc_= []
    for i in range(0,41-num_class): 
        face_10_subjs = np.array(Img[10*i:10*(i+num_class),:])
        face_10_subjs = face_10_subjs.astype(float)        
        label_10_subjs = np.array(Label[10*i:10*(i+num_class)]) 
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs) 
                     
        CAE.initlization()        
        CAE.restore() # restore from pre-trained model    
        
        max_step = 900#50 + num_class*25# 100+num_class*20
        display_step = max_step#10
        lr = 1.0e-3
        # fine-tune network
        epoch = 0
        while epoch < max_step:
            epoch = epoch + 1           
            cost, Coef = CAE.partial_fit(face_10_subjs, lr)#                                  
            if epoch % display_step == 0:
                print("epoch: %.1d" % epoch, "cost: %.8f" % (cost/float(batch_size)))                
                Coef = thrC(Coef,alpha)                                                       
                y_x, CKsym = post_proC(Coef, label_10_subjs.max(), 3,1)                  
                missrate_x = err_rate(label_10_subjs, y_x)                
                acc_x = 1 - missrate_x 
                print("experiment: %d" % i, "our accuracy: %.4f" % acc_x)
        acc_.append(acc_x)    
    
    acc_ = np.array(acc_)
    m = np.mean(acc_)
    me = np.median(acc_)
    print("%d subjects:" % num_class)    
    print("Mean: %.4f%%" % ((1-m)*100))    
    sio.savemat('2WORL',{'W1':Coef,'W2':CKsym})
    print(acc_) 
    
    return (1-m), (1-me)
    
   
        
    

    
# load face images and labels
data = sio.loadmat('ORL_32x32.mat')
Img = data['fea']
Label = data['gnd']     
graph = sio.loadmat('ORL_graph.mat')
L = graph['L']
# L = np.zeros([len(Img),len(Img)])
# L = L.astype(np.float32)

# face image clustering
n_input = [32, 32]
kernel_size = [3,3,3]
n_hidden = [3, 3, 5]

Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1],1]) 

all_subjects = [40]

avg = []
med = []


num_class = all_subjects[0]
batch_size = num_class * 10
reg1 = 1.0
reg2 = 10.0
reg3 = 8
max_step = 1550#50 + num_class*25# 100+num_class*20
display_step = 50#10
lr = 1.0e-3
alpha = 0.17
 
model_path = './pretrain-model-ORL/model-335-32x32-orl.ckpt' 
restore_path = './pretrain-model-ORL/model-335-32x32-orl.ckpt' 
logs_path = './pretrain-model-ORL/logs' 
tf.reset_default_graph()
CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, reg3 = reg3, L = L,\
             kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, restore_path=restore_path, logs_path=logs_path)





print(alpha)

acc_= []
# for i in range(0,41-num_class): 
i = 0
face_10_subjs = np.array(Img[10*i:10*(i+num_class),:])
face_10_subjs = face_10_subjs.astype(float)        
label_10_subjs = np.array(Label[10*i:10*(i+num_class)]) 
label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
label_10_subjs = np.squeeze(label_10_subjs)
             
CAE.initlization()        
CAE.restore() # restore from pre-trained model    

epoch = 0
while epoch < max_step:
    epoch = epoch + 1           
    cost, Coef = CAE.partial_fit(face_10_subjs, lr)#                                  
    if epoch % display_step == 0:
        print("epoch: %.1d" % epoch, "cost: %.8f" % (cost/float(batch_size)))                
        Coef = thrC(Coef,alpha)                                                       
        y_x, CKsym = post_proC(Coef, label_10_subjs.max(),4,2)                  
        missrate_x = err_rate(label_10_subjs, y_x)                
        acc_x = 1 - missrate_x 
        print("experiment: %d" % i, "our accuracy: %.4f" % acc_x)
        acc_.append(acc_x)    

# embed = CAE.transform(face_10_subjs)
acc_ = np.array(acc_)
m = np.mean(acc_)
me = np.median(acc_)
print("%d subjects:" % num_class)    
print("Mean: %.4f%%" % ((1-m)*100))    
# sio.savemat('orlem2',{'x':embed,'Y':label_10_subjs})
print(acc_.max())