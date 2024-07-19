#active learning
from scipy.optimize import fsolve
from pylab import plt
#plt.style.use('seaborn')
#import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 15})
import numpy as np
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import dill
import tensorflow as tf
import time
import argparse

parser = argparse.ArgumentParser(description='Solve the 4D model')
parser.add_argument('-l','--load',type=str,metavar='',nargs='?',default='no',
                    help='Specify whether to load value fucntion from input file')
parser.add_argument('-s','--save',type=str, metavar='',nargs='?',default='no',
                    help='Specify whether to save pickle (warning: requires ~1GB space)')
parser.add_argument('-act','--active',type=str, metavar='',nargs='?',default='no',
                    help='Specify whether to use active learning')
parser.add_argument('-epe','--epochsExperts',type=int,
                    nargs='?',default=5000,help='Specify number of epochs for experts')
parser.add_argument('-eph','--epochsHouseholds',type=int,
                    nargs='?',default=5000,help='Specify number of epochs for households')
args = parser.parse_args()


class nnpde_informed():
    def __init__(self,linearTerm,advection_z,advection_g,advection_s,advection_a,diffusion_z,diffusion_g,diffusion_s,diffusion_a,
                    cross_term_zg,cross_term_zs,cross_term_za,J0,X,layers,X_f,dt,tb,learning_rate,maxEpochs):
        
        self.linearTerm = linearTerm
        self.advection_z = advection_z
        self.advection_y = advection_g
        self.advection_s = advection_s
        self.advection_a = advection_a
        self.diffusion_z = diffusion_z
        self.diffusion_y = diffusion_g
        self.diffusion_s = diffusion_s
        self.diffusion_a = diffusion_a
        self.cross_term_zs = cross_term_zs
        self.cross_term_zy = cross_term_zg
        self.cross_term_za = cross_term_za
        self.u = J0
        self.X = X
        self.layers = layers
        self.t_b = tb
        self.X_f = X_f
        self.dt = dt
        self.learning_rate = learning_rate
        self.maxEpochs = maxEpochs
        
        
        self.z_u = self.X[:,0:1]
        self.y_u = self.X[:,1:2]
        self.s_u = self.X[:,2:3]
        self.a_u = self.X[:,3:4]
        self.t_u = self.X[:,4:5]
        
        self.z_f = self.X_f[:,0:1]
        self.y_f = self.X_f[:,1:2]
        self.s_f = self.X_f[:,2:3]
        self.a_f = self.X_f[:,3:4]
        self.t_f = self.X_f[:,4:5]    

        self.lb = np.array([0,self.y_u[0][0], self.s_u[0][0], self.a_u[0][0],self.dt])
        self.ub = np.array([1,self.y_u[-1][0], self.s_u[-1][0], self.a_u[-1][0], 0])
        #self.lb = 0
        #self.ub = 1
        
        self.X_b = np.array([[self.z_u[0][0],self.y_u[0][0],self.s_u[0][0],self.a_u[0][0], 0],[self.z_u[0][0],self.y_u[0][0],self.s_u[0][0],self.a_u[0][0], self.dt],[self.z_u[-1][0],self.y_u[-1][0],self.s_u[-1][0],self.a_u[-1][0],0.],[self.z_u[-1][0],self.y_u[-1][0],self.s_u[-1][0],self.a_u[-1][0],self.dt]])
        self.z_b = np.array(self.X_b[:,0]).reshape(-1,1)
        self.y_b = np.array(self.X_b[:,1]).reshape(-1,1)
        self.s_b = np.array(self.X_b[:,2]).reshape(-1,1)
        self.a_b = np.array(self.X_b[:,3]).reshape(-1,1)
        self.t_b = np.array(self.X_b[:,4]).reshape(-1,1)
        #Initialize NNs
        self.weights, self.biases = self.initialize_nn(layers)
        
        #tf placeholders and computational graph
        self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True))
        self.z_u_tf = tf.placeholder(tf.float32,shape=[None,self.z_u.shape[1]])
        self.y_u_tf = tf.placeholder(tf.float32,shape=[None,self.y_u.shape[1]])
        self.s_u_tf = tf.placeholder(tf.float32,shape=[None,self.s_u.shape[1]])
        self.a_u_tf = tf.placeholder(tf.float32,shape=[None,self.a_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32,shape=[None,self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None,self.u.shape[1]])
        
        
        self.z_b_tf =  tf.placeholder(tf.float32, shape=[None,self.z_b.shape[1]])
        self.y_b_tf =  tf.placeholder(tf.float32, shape=[None,self.y_b.shape[1]])
        self.s_b_tf =  tf.placeholder(tf.float32, shape=[None,self.s_b.shape[1]])
        self.a_b_tf =  tf.placeholder(tf.float32, shape=[None,self.a_b.shape[1]])
        self.t_b_tf =  tf.placeholder(tf.float32, shape=[None,self.t_b.shape[1]])
        
        self.z_f_tf = tf.placeholder(tf.float32, shape=[None,self.z_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None,self.y_f.shape[1]])
        self.s_f_tf = tf.placeholder(tf.float32, shape=[None,self.s_f.shape[1]])
        self.a_f_tf = tf.placeholder(tf.float32, shape=[None,self.a_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None,self.t_f.shape[1]])
        
        
        self.u_pred,_,_,_,_ = self.net_u(self.z_u_tf,self.y_u_tf,self.s_u_tf,self.a_u_tf,self.t_u_tf)
        self.f_pred = self.net_f(self.z_f_tf, self.y_f_tf, self.s_f_tf,self.a_f_tf,self.t_f_tf)
        _, self.ub_z_pred, self.ub_y_pred,self.ub_s_pred,self.ub_a_pred = self.net_u(self.z_b_tf,self.y_b_tf,self.s_b_tf,self.a_b_tf,self.t_b_tf)
        
        
        
        self.loss = tf.reduce_mean(tf.square(self.u_tf-self.u_pred)) + \
                        tf.reduce_mean(tf.square(self.f_pred))  #+\
                        #tf.reduce_mean(tf.square(self.ub_y_pred)) +\
                        #tf.reduce_mean(tf.square(self.ub_z_pred)) +\
                        #tf.reduce_mean(tf.square(self.ub_s_pred))
                        
                        
                        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,method='L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol': 1e-08})
                                                                           
        
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    def initialize_nn(self,layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers-1):
            W = self.xavier_init(size = [layers[l],layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype = tf.float32)
            weights.append(W)
            biases.append(b)
        
        return weights,biases

    
    def xavier_init(self,size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        try:
            val = tf.Variable(tf.random.truncated_normal([in_dim,out_dim], stddev = xavier_stddev), dtype = tf.float32)
        except:
            val = tf.Variable(tf.truncated_normal([in_dim,out_dim], stddev = xavier_stddev), dtype = tf.float32)
        return val
    
    def neural_net(self,X,weights,biases):
        num_layers = len(weights) +1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) -1
        #H=X
        for l in range(num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H,W),b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.sigmoid(tf.add(tf.matmul(H,W),b))
        return Y

    def net_u(self,z,y,s,a,t):
        X = tf.concat([z,y,s,a,t],1)
        u = self.neural_net(X,self.weights,self.biases)
        u_z = tf.gradients(u,z)[0]
        u_y = tf.gradients(u,y)[0]
        u_s = tf.gradients(u,s)[0]
        u_a = tf.gradients(u,a)[0]
        return u,u_z,u_y,u_s,u_a
    
    def net_f(self,z,y,s,a,t):
        u,u_z,u_y,u_s,u_a = self.net_u(z,y,s,a,t)
        u_t = tf.gradients(u,t)[0]
        u_zz = tf.gradients(u_z,z)[0]
        u_yy = tf.gradients(u_y,y)[0]
        u_zy = tf.gradients(u_z,y)[0]
        u_ss = tf.gradients(u_s,s)[0]
        u_aa = tf.gradients(u_a,a)[0]
        u_zs = tf.gradients(u_z,s)[0]
        u_za = tf.gradients(u_z,a)[0]
        f =  u_t + self.diffusion_z * u_zz +  self.diffusion_y * u_yy + self.diffusion_s * u_ss +self.diffusion_a * u_aa + \
                 self.advection_y * u_y + self.advection_z * u_z + self.advection_s * u_s+self.advection_a * u_a+ \
                 self.cross_term_zy * u_zy + self.cross_term_zs * u_zs + self.cross_term_za * u_za -  self.linearTerm *u
        return f
    
    def callback(self,loss):
        print('Loss: ',loss)
    
    def train(self):
        #K.clear_session()
        tf_dict = {self.z_u_tf: self.z_u, self.y_u_tf: self.y_u, self.s_u_tf: self.s_u,self.a_u_tf: self.a_u, self.t_u_tf: self.t_u, self.u_tf:self.u,
                    self.z_f_tf: self.z_f,self.y_f_tf: self.y_f, self.s_f_tf: self.s_f,self.a_f_tf: self.a_f, self.t_f_tf: self.t_f,
                    self.z_b_tf: self.z_b,
                    self.y_b_tf: self.y_b,
                    self.s_b_tf: self.s_b,
                    self.a_b_tf: self.a_b,
                    self.t_b_tf: self.t_b}
                 
        start_time = time.time()
        
        if True: #set this to true if you want adam to run 
            for it in range(self.maxEpochs):
                self.sess.run(self.train_op_Adam, tf_dict)
                # Print
                if it % 1000 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' % 
                          (it, loss_value, elapsed))
                    start_time = time.time()
        
            start_time = time.time()
        self.optimizer.minimize(self.sess,feed_dict = tf_dict)
        elapsed = time.time() - start_time
        print('Time: %.2f' % elapsed)
        #self.sess.close()


    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.z_u_tf: X_star[:,0:1],self.y_u_tf: X_star[:,1:2],self.s_u_tf: X_star[:,2:3],self.a_u_tf: X_star[:,3:4], self.t_u_tf: X_star[:,4:5]})
        tf.reset_default_graph()
        return u_star


class model4D():
    def __init__(self,params):
        self.params = params        
        self.Nz = 300
        self.Ng = 21
        self.Ns = 22
        self.Na = 23
        self.z = np.linspace(0.001,0.999, self.Nz)
        self.dz  = self.z[1:self.Nz] - self.z[0:self.Nz-1];  
        self.dz2 = self.dz[0:self.Nz-2]**2;
        self.z_mat = np.tile(self.z.reshape(self.Nz,1,1,1),(1,self.Ng,self.Ns,self.Na))
        self.dz_mat = np.tile(self.dz.reshape(self.Nz-1,1,1,1),(1,self.Ng,self.Ns,self.Na))
        self.dz2_mat = np.tile(self.dz2.reshape(self.Nz-2,1,1,1),(1,self.Ng,self.Ns,self.Na))
        
        self.g = np.linspace(self.params['g_l'], self.params['g_u'], self.Ng)
        self.dg = self.g[1:self.Ng] - self.g[0:self.Ng-1]
        self.dg2 = self.dg[0:self.Ng-2]**2
        self.g_mat = np.tile(self.g.reshape(1,self.Ng,1,1),(self.Nz,1,self.Ns,self.Na))
        self.dg_mat = np.tile(self.dg.reshape(1,self.Ng-1,1,1),(self.Nz,1,self.Ns,self.Na))
        self.dg2_mat = np.tile(self.dg2.reshape(1,self.Ng-2,1,1),(self.Nz,1,self.Ns,self.Na))

        self.s = np.linspace(self.params['s_l'], self.params['s_u'], self.Ns)
        self.ds = self.s[1:self.Ns] - self.s[0:self.Ns-1]
        self.ds2 = self.ds[0:self.Ns-2]**2
        self.s_mat = np.tile(self.s.reshape(1,1,self.Ns,1),(self.Nz,self.Ng,1,self.Na))
        self.ds_mat = np.tile(self.ds.reshape(1,1,self.Ns-1,1),(self.Nz,self.Ng,1,self.Na))
        self.ds2_mat = np.tile(self.ds2.reshape(1,1,self.Ns-2,1),(self.Nz,self.Ng,1,self.Na))
        
        self.a = np.linspace(self.params['a_l'], self.params['a_u'], self.Na)
        self.da = self.a[1:self.Na] - self.a[0:self.Na-1]
        self.da2 = self.da[0:self.Na-2]**2
        self.a_mat = np.tile(self.a.reshape(1,1,1,self.Na),(self.Nz,self.Ng,self.Ns,1))
        self.da_mat = np.tile(self.da.reshape(1,1,1,self.Na-1),(self.Nz,self.Ng,self.Ns,1))
        self.da2_mat = np.tile(self.da2.reshape(1,1,1,self.Na-2),(self.Nz,self.Ng,self.Ns,1))
        
        self.q   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.qz   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.qzz   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.qs   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.qss   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.qa   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.qaa   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.qg   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.qgg   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.Qsz   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.Qsg  =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.Qzg   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.Qza   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));

        self.thetah   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.theta   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.r   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.ssq   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.ssg   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.sss   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.ssa   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.iota   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        self.chi   =  np.zeros((self.Nz,self.Ng,self.Ns,self.Na));
        
        if params['load']=='yes':
            print('loading value functions from disc')
            self.Je = np.load('4Dmodel_Je.npy')
            self.Jh = np.load('4Dmodel_Jh.npy')
            self.amax_vec = list(np.load('4Dmodel_error.npy'))
        else:
            print('initializing arbitrary value functions')
            self.Je = np.ones([self.Nz,self.Ng,self.Ns,self.Na]) * 1
            self.Jh = np.ones([self.Nz,self.Ng, self.Ns,self.Na]) * 1
            self.amax_vec=[]
        self.crisis = np.zeros((self.Ng, self.Ns,self.Na))
        
        self.Jtilde_z= np.zeros((self.Nz,self.Ng,self.Ns,self.Na))
        self.Jtilde_f= np.zeros((self.Nz,self.Ng,self.Ns,self.Na)) 

        self.first_time = np.zeros((self.Ng,self.Ns,self.Na))
        self.psi = np.zeros((self.Nz,self.Ng,self.Ns,self.Na))
        
        self.maxIterations=40
        self.convergenceCriterion = 1e-2;
        self.converged = False
        self.Iter=0
        try:
            if not os.path.exists('../output'):
                os.mkdir('../output')
        except:
            print('Warning: Not possible to plot. You are either in Colab or gpu cluster')
        self.amax = np.float('Inf')


        with open('4Dmodel_params.txt', 'w') as f:
            print(self.params, file=f)
        
    def equations_region1(self,q_p, Psi_p, sig_qk_p, sig_qg_p, sig_qs_p, sig_qa_p, zi, gi, si,ai):
        i_p = (q_p - 1)/self.params['kappa']
        eq1 = (self.a_mat[zi,gi,si,ai]-self.params['aH'])/q_p -\
                self.params['alpha'] * self.Jtilde_z[zi,gi,si,ai]*(self.params['alpha'] * Psi_p - self.z_mat[zi,gi,si,ai])*(sig_qk_p**2 + sig_qg_p**2 + sig_qs_p**2) - self.Jtilde_g[zi,gi,si,ai]*self.sig_g[zi,gi,si,ai]*sig_qg_p - self.Jtilde_s[zi,gi,si,ai]*self.sig_s[zi,gi,si,ai]*sig_qs_p -self.Jtilde_a[zi,gi,si,ai]*self.sig_a[zi,gi,si,ai]*sig_qa_p -\
                self.params['sigma'] * np.sqrt(self.s_mat[zi,gi,si,ai]) *sig_qk_p*(self.params['gamma'] - self.params['gamma'])
        eq2 = (self.params['rho']*self.z_mat[zi,gi,si,ai] + self.params['rho']*(1-self.z_mat[zi,gi,si,ai])) * q_p  - Psi_p * (self.a[ai] - i_p) - (1- Psi_p) * (self.params['aH'] - i_p)
              
        eq3 = sig_qk_p - sig_qk_p*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])/self.dz[zi-1] + (sig_qk_p)*self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,gi,si,ai]) - self.params['sigma']* np.sqrt(self.s_mat[zi,gi,si,ai])

        if gi==0:
            eq4 = sig_qg_p * self.q[zi-1,gi,si,ai]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,gi,si,ai]) - self.sig_g[zi,gi,si,ai]/self.dg[gi-1]  + sig_qg_p - sig_qg_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])
        else:
            eq4 = sig_qg_p * self.q[zi-1,gi,si,ai]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,gi,si,ai]) - self.sig_g[zi,gi,si,ai]/self.dg[gi-1] + self.sig_g[zi,gi,si,ai] * self.q[zi,gi-1,si,ai]/(q_p * self.dg[gi-1]) + sig_qg_p - sig_qg_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])
        
        if si==0:
            eq5 = sig_qs_p * self.q[zi-1,gi,si,ai]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,gi,si,ai]) - self.sig_s[zi,gi,si,ai]/self.ds[si-1]  + sig_qs_p - sig_qs_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])
        else:
            eq5 = sig_qs_p * self.q[zi-1,gi,si,ai]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,gi,si,ai]) - self.sig_s[zi,gi,si,ai]/self.ds[si-1] + self.sig_s[zi,gi,si,ai] * self.q[zi,gi,si-1,ai]/(q_p * self.ds[si-1]) + sig_qs_p - sig_qs_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])

        if ai==0:
            eq6 = sig_qa_p * self.q[zi-1,gi,si,ai]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,gi,si,ai]) - self.sig_a[zi,gi,si,ai]/self.da[ai-1]  + sig_qa_p - sig_qa_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])
        else:
            eq6 = sig_qa_p * self.q[zi-1,gi,si,ai]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,gi,si,ai]) - self.sig_a[zi,gi,si,ai]/self.da[ai-1] + self.sig_a[zi,gi,si,ai] * self.q[zi,gi,si,ai-1]/(q_p * self.da[ai-1]) + sig_qa_p - sig_qa_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])


        ER = np.array([eq1,eq2,eq3,eq4,eq5,eq6])
        QN = np.zeros(shape=(6,6))

        QN[0,:] = np.array([-self.params['alpha']**2 * self.Jtilde_z[zi,gi,si,ai]*(sig_qk_p**2 + sig_qg_p**2+ sig_qs_p**2), -2*self.Jtilde_z[zi,gi,si,ai]*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai])*sig_qk_p - self.params['sigma']*  np.sqrt(self.s_mat[zi,gi,si,ai]) * (self.params['gamma']-self.params['gamma']), \
                            -2* self.params['alpha'] * self.Jtilde_z[zi,gi,si,ai]*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])*sig_qg_p - self.Jtilde_g[zi,gi,si,ai]*self.sig_g[zi,gi,si,ai],
                            -2* self.params['alpha'] * self.Jtilde_z[zi,gi,si,ai]*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])*sig_qs_p - self.Jtilde_s[zi,gi,si,ai]*self.sig_s[zi,gi,si,ai],
                            -2* self.params['alpha'] * self.Jtilde_z[zi,gi,si,ai]*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])*sig_qa_p - self.Jtilde_a[zi,gi,si,ai]*self.sig_a[zi,gi,si,ai], -(self.a_mat[zi,gi,si,ai]-self.params['aH'])/(q_p**2)])
        QN[1,:] = np.array([self.params['aH'] - self.a_mat[zi,gi,si,ai], 0, 0,0, 0, self.params['rho'] * self.z_mat[zi,gi,si,ai] + (1-self.z_mat[zi,gi,si,ai])*self.params['rho'] + 1/self.params['kappa']])
        QN[2,:] = np.array([-sig_qk_p * self.params['alpha']/self.dz[zi-1]*(1-self.q[zi-1,gi,si,ai]/q_p), 1-((self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])/self.dz[zi-1])*(q_p - self.q[zi-1,gi,si,ai])/q_p, \
                                 0, 0,0, -sig_qk_p*(self.q[zi-1,gi,si,ai]/q_p**2)*(self.params['alpha'] * Psi_p-self.z_mat[zi,gi,si,ai])/self.dz[zi-1]])
        if gi==0:
            QN[3,:] = np.array([-sig_qg_p/self.dz[zi-1] + sig_qg_p/self.dz[zi-1] * self.q[zi-1,gi,si,ai]/q_p, 0, 1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) ,0, 0, -sig_qg_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai])])
        else:
            QN[3,:] = np.array([-sig_qg_p/self.dz[zi-1] + sig_qg_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/q_p, 0, \
                                1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]),0,0, -sig_qg_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) - self.sig_g[zi,gi,si,ai]*self.q[zi,gi-1,si,ai]/(q_p**2 * self.dg[gi-1]) ])
        if si==0:
            QN[4,:] = np.array([-sig_qs_p/self.dz[zi-1] + sig_qs_p/self.dz[zi-1] * self.q[zi-1,gi,si,ai]/q_p, 0,0, 1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) ,0, -sig_qs_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai])])
        else:
            QN[4,:] = np.array([-sig_qs_p/self.dz[zi-1] + sig_qs_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/q_p, 0,0, \
                                1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]), 0,-sig_qs_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) - self.sig_s[zi,gi,si,ai]*self.q[zi,gi,si-1,ai]/(q_p**2 * self.ds[si-1]) ])
    
        if ai==0:
            QN[5,:] = np.array([-sig_qa_p/self.dz[zi-1] + sig_qa_p/self.dz[zi-1] * self.q[zi-1,gi,si,ai]/q_p, 0,0,0, 1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) , -sig_qa_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai])])
        else:
            QN[5,:] = np.array([-sig_qs_p/self.dz[zi-1] + sig_qs_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/q_p, 0,0,0, \
                                1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]),-sig_qa_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,gi,si,ai]) - self.sig_a[zi,gi,si,ai]*self.q[zi,gi,si,ai-1]/(q_p**2 * self.da[ai-1]) ])

        EN = np.array([Psi_p, sig_qk_p, sig_qg_p, sig_qs_p, sig_qa_p,q_p]) - np.linalg.solve(QN,ER)
        del ER, QN
        return EN

    def equations_region2(self,q_p,sig_qk_p,sig_qg_p, sig_qs_p, sig_qa_p, zi,gi,si,ai):
        i_p = (q_p-1)/self.params['kappa']
        eq1 = (self.params['rho']*self.z_mat[zi,gi,si,ai] + self.params['rho']*(1-self.z_mat[zi,gi,si,ai])) * q_p  - (self.a_mat[zi,gi,si,ai] - i_p)
        eq2 = sig_qk_p - sig_qk_p*(1-self.z_mat[zi,gi,si,ai])/self.dz[zi-1] + (sig_qk_p)*self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1 - self.z_mat[zi,gi,si,ai]) - self.params['sigma'] * np.sqrt(self.s_mat[zi,gi,si,ai])

        if gi==0:
            eq3 = sig_qg_p*self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]) - self.sig_g[zi,gi,si,ai]/self.dg[gi-1]  + sig_qg_p - sig_qg_p/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai])
        else:
            eq3 = sig_qg_p*self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]) - self.sig_g[zi,gi,si,ai]/self.dg[gi-1] + self.sig_g[zi,gi,si,ai]*self.q[zi,gi-1,si,ai]/(q_p*self.dg[gi-1]) + sig_qg_p - sig_qg_p/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai])
        
        if si==0:
            eq4 = sig_qs_p*self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]) - self.sig_s[zi,gi,si,ai]/self.ds[si-1]  + sig_qs_p - sig_qs_p/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai])
        else:
            eq4 = sig_qs_p*self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]) - self.sig_s[zi,gi,si,ai]/self.ds[si-1] + self.sig_s[zi,gi,si,ai]*self.q[zi,gi,si-1,ai]/(q_p*self.ds[si-1]) + sig_qs_p - sig_qs_p/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai])
        
        if ai==0:
            eq5 = sig_qa_p*self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]) - self.sig_a[zi,gi,si,ai]/self.da[ai-1]  + sig_qa_p - sig_qa_p/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai])
        else:
            eq5 = sig_qa_p*self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]) - self.sig_a[zi,gi,si,ai]/self.da[ai-1] + self.sig_a[zi,gi,si,ai]*self.q[zi,gi,si,ai-1]/(q_p*self.da[ai-1]) + sig_qa_p - sig_qa_p/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai])
        
        ER = np.array([eq1,eq2,eq3,eq4,eq5])
        QN = np.zeros(shape=(5,5))
        QN[0,:] = np.array([0,0,0,0,self.params['rho']*self.z_mat[zi,gi,si,ai] + (1-self.z_mat[zi,gi,si,ai])*self.params['rho'] + 1/self.params['kappa']])
        QN[1,:] = np.array([1-(1-self.z_mat[zi,gi,si,ai])/self.dz[zi-1] + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]), 0, 0,0, -sig_qk_p*(self.q[zi-1,gi,si,ai]/q_p**2)*(1-self.z_mat[zi,gi,si,ai])/self.dz[zi-1]])
        if gi==0:
            QN[2,:] = np.array([0,1-1/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]), 0,0, -sig_qg_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(1-self.z_mat[zi,gi,si,ai])])
        else:
            QN[2,:] = np.array([0,1-1/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]), 0,0, -sig_qg_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(1-self.z_mat[zi,gi,si,ai]) - self.sig_g[zi,gi-1,si,ai]*self.q[zi,gi-1,si,ai]/(q_p**2 * self.dg[gi-1])])
        if si==0:
            QN[3,:] = np.array([0,0,1-1/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]),0, -sig_qs_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(1-self.z_mat[zi,gi,si,ai])])
        else:
            QN[3,:] = np.array([0,0,1-1/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]),0, -sig_qs_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(1-self.z_mat[zi,gi,si,ai]) - self.sig_s[zi,gi,si-1,ai]*self.q[zi,gi,si-1,ai]/(q_p**2 * self.ds[si-1])])
        
        if ai==0:
            QN[4,:] = np.array([0,0,0,1-1/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]), -sig_qa_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(1-self.z_mat[zi,gi,si,ai])])
        else:
            QN[4,:] = np.array([0,0,0,1-1/self.dz[zi-1]*(1-self.z_mat[zi,gi,si,ai]) + self.q[zi-1,gi,si,ai]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,gi,si,ai]), -sig_qa_p/self.dz[zi-1]*self.q[zi-1,gi,si,ai]/(q_p**2)*(1-self.z_mat[zi,gi,si,ai]) - self.sig_a[zi,gi,si,ai-1]*self.q[zi,gi,si,ai-1]/(q_p**2 * self.da[ai-1])])
      
        EN = np.array([sig_qk_p,sig_qg_p,sig_qs_p,sig_qa_p,q_p]) - np.linalg.solve(QN,ER)
        del ER,QN
        return EN
    
    def pickle_stuff(self,object_name,filename):
                    with open(filename,'wb') as f:
                        dill.dump(object_name,f)
    def solve(self,pde='True'):
        self.psi[0,:,:,:]=0
        self.q[0,:,:,:] = (1 + self.params['kappa']*(self.params['aH'] + self.psi[0,:]*(self.a_mat[0,:]-self.params['aH'])))/(1 + self.params['kappa']*(self.params['rho'] + self.z[0] * (self.params['rho'] - self.params['rho'])));
        self.chi[0,:,:,:] = 0;
        self.ssq = self.params['sigma'] * np.sqrt(self.s_mat);
        self.q0 = (1 + self.params['kappa'] * self.params['aH'])/(1 + self.params['kappa'] * self.params['rho']); 
        self.iota[0,:,:,:] = (self.q0-1)/self.params['kappa']
        self.sig_g =  self.params['beta_g']*(self.params['g_u'] - self.g_mat)*(self.g_mat - self.params['g_l'])
        self.sig_s = self.params['beta_s']*(self.params['s_u'] - self.s_mat) * (self.s_mat - self.params['s_l'])
        self.sig_a = self.params['beta_a']*(self.params['a_u'] - self.a_mat) * (self.a_mat - self.params['a_l'])
        self.ssg = np.zeros((self.Nz,self.Ng,self.Ns,self.Na))
        self.sss = np.zeros((self.Nz,self.Ng,self.Ns,self.Na))
        self.ssa = np.zeros((self.Nz,self.Ng,self.Ns,self.Na))

        for timeStep in range(self.maxIterations):
            self.Iter+=1            
            self.crisis_eta = 0;
            self.logValueE = np.log(self.Je);
            self.logValueH = np.log(self.Jh);
            self.dLogJe_z = np.vstack([((self.logValueE[1,:,:,:]-self.logValueE[0,:,:,:])/(self.z_mat[1,:,:,:]-self.z_mat[0,:,:,:])).reshape(-1,self.Ng,self.Ns,self.Na),(self.logValueE[2:,:,:,:]-self.logValueE[0:-2,:,:,:])/(self.z_mat[2:,:,:,:]-self.z_mat[0:-2,:,:,:]),((self.logValueE[-1,:,:,:]-self.logValueE[-2,:,:,:])/(self.z_mat[-1,:,:,:]-self.z_mat[-2,:,:,:])).reshape(-1,self.Ng,self.Ns,self.Na)]);
            self.dLogJh_z = np.vstack([((self.logValueH[1,:,:,:]-self.logValueH[0,:,:,:])/(self.z_mat[1,:,:,:]-self.z_mat[0,:,:,:])).reshape(-1,self.Ng,self.Ns,self.Na),(self.logValueH[2:,:,:,:]-self.logValueH[0:-2,:,:,:])/(self.z_mat[2:,:,:,:]-self.z_mat[0:-2,:,:,:]),((self.logValueH[-1,:,:,:]-self.logValueH[-2,:,:,:])/(self.z_mat[-1,:,:,:]-self.z_mat[-2,:,:,:])).reshape(-1,self.Ng,self.Ns,self.Na)]);
            self.dLogJe_g = np.hstack([((self.logValueE[:,1,:,:]-self.logValueE[:,0,:,:])/(self.g_mat[:,1,:,:]-self.g_mat[:,0,:,:])).reshape(self.Nz,-1,self.Ns,self.Na),(self.logValueE[:,2:,:,:]-self.logValueE[:,0:-2,:,:])/(self.g_mat[:,2:,:,:]-self.g_mat[:,0:-2,:,:]),((self.logValueE[:,-1,:,:]-self.logValueE[:,-2,:,:])/(self.g_mat[:,-1,:,:]-self.g_mat[:,-2,:,:])).reshape(self.Nz,-1,self.Ns,self.Na)]);
            self.dLogJh_g = np.hstack([((self.logValueH[:,1,:,:]-self.logValueH[:,0,:,:])/(self.g_mat[:,1,:,:]-self.g_mat[:,0,:,:])).reshape(self.Nz,-1,self.Ns,self.Na),(self.logValueH[:,2:,:,:]-self.logValueH[:,0:-2,:,:])/(self.g_mat[:,2:,:,:]-self.g_mat[:,0:-2,:,:]),((self.logValueH[:,-1,:,:]-self.logValueH[:,-2,:,:])/(self.g_mat[:,-1,:,:]-self.g_mat[:,-2,:,:])).reshape(self.Nz,-1,self.Ns,self.Na)]);
            self.dLogJe_s = np.dstack([((self.logValueE[:,:,1,:]-self.logValueE[:,:,0,:])/(self.s_mat[:,:,1,:]-self.s_mat[:,:,0,:])).reshape(self.Nz,self.Ng,-1,self.Na),
                                      (self.logValueE[:,:,2:,:]-self.logValueE[:,:,0:-2,:])/(self.s_mat[:,:,2:,:]-self.s_mat[:,:,0:-2,:]),
                                      ((self.logValueE[:,:,-1,:]-self.logValueE[:,:,-2,:])/(self.s_mat[:,:,-1,:]-self.s_mat[:,:,-2,:])).reshape(self.Nz,self.Ng,-1,self.Na)]);
            self.dLogJh_s = np.dstack([((self.logValueH[:,:,1,:]-self.logValueH[:,:,0,:])/(self.s_mat[:,:,1,:]-self.s_mat[:,:,0,:])).reshape(self.Nz,self.Ng,-1,self.Na),
                                       (self.logValueH[:,:,2:,:]-self.logValueH[:,:,0:-2,:])/(self.s_mat[:,:,2:,:]-self.s_mat[:,:,0:-2,:]),
                                       ((self.logValueH[:,:,-1,:]-self.logValueH[:,:,-2,:])/(self.s_mat[:,:,-1,:]-self.s_mat[:,:,-2,:])).reshape(self.Nz,self.Ng,-1,self.Na)]);
            
            self.dLogJe_a = np.concatenate([((self.logValueE[:,:,:,1]-self.logValueE[:,:,:,0])/(self.a_mat[:,:,:,1]-self.a_mat[:,:,:,0])).reshape(self.Nz,self.Ng,self.Ns,-1),
                                      (self.logValueE[:,:,:,2:]-self.logValueE[:,:,:,0:-2])/(self.a_mat[:,:,:,2:]-self.a_mat[:,:,:,0:-2]),
                                      ((self.logValueE[:,:,:,-1]-self.logValueE[:,:,:,-2])/(self.a_mat[:,:,:,-1]-self.a_mat[:,:,:,-2])).reshape(self.Nz,self.Ng,self.Ns,-1)],axis=3);
            self.dLogJh_a = np.concatenate([((self.logValueH[:,:,:,1]-self.logValueH[:,:,:,0])/(self.a_mat[:,:,:,1]-self.a_mat[:,:,:,0])).reshape(self.Nz,self.Ng,self.Ns,-1),
                                       (self.logValueH[:,:,:,2:]-self.logValueH[:,:,:,0:-2])/(self.a_mat[:,:,:,2:]-self.a_mat[:,:,:,0:-2]),
                                       ((self.logValueH[:,:,:,-1]-self.logValueH[:,:,:,-2])/(self.a_mat[:,:,:,-1]-self.a_mat[:,:,:,-2])).reshape(self.Nz,self.Ng,self.Ns,-1)],axis=3);
                        
            if self.params['scale']>1:    
                self.Jtilde_z = (1-self.params['gamma'])*self.dLogJh_z - (1-self.params['gamma'])*self.dLogJe_z + 1/(self.z_mat*(1-self.z_mat))
                self.Jtilde_g = (1-self.params['gamma'])*self.dLogJh_g - (1-self.params['gamma'])*self.dLogJe_g
                self.Jtilde_s = (1-self.params['gamma'])*self.dLogJh_s - (1-self.params['gamma'])*self.dLogJe_s
                self.Jtilde_a = (1-self.params['gamma'])*self.dLogJh_a - (1-self.params['gamma'])*self.dLogJe_a
            else:
                self.Jtilde_z = self.dLogJh_z - self.dLogJe_z + 1/(self.z_mat*(1-self.z_mat))
                self.Jtilde_g = self.dLogJh_g - self.dLogJe_g
                self.Jtilde_s = self.dLogJh_s - self.dLogJe_s
                self.Jtilde_a = self.dLogJh_a - self.dLogJe_a
            
            #gi=0;si=0;zi=1
            for gi in range(self.Ng):
                for si in range(self.Ns):
                    for ai in range(0,self.Na):
                        for zi in range(1,self.Nz):
                            if self.psi[zi-1,gi,si,ai]<1:
                                result= self.equations_region1(self.q[zi-1,gi,si,ai], self.psi[zi-1,gi,si,ai], self.ssq[zi-1,gi,si,ai], self.ssg[zi-1,gi,si,ai],self.sss[zi-1,gi,si,ai],self.ssa[zi-1,gi,si,ai], zi, gi,si,ai)
                                if result[0]>=1:
                                    #break
                                    self.crisis[gi,si,ai]=zi
                                    self.psi[zi,gi,si,ai]=1
                                    self.chi[zi,gi,si,ai] = np.maximum(self.z[zi],self.params['alpha'])
                                    result = self.equations_region2(self.q[zi-1,gi,si,ai],self.ssq[zi-1,gi,si,ai],self.ssg[zi-1,gi,si,ai],self.sss[zi-1,gi,si,ai],self.ssa[zi-1,gi,si,ai],zi,gi,si,ai)
                                    self.ssq[zi,gi,si,ai], self.ssg[zi,gi,si,ai],self.sss[zi,gi,si,ai],self.ssa[zi,gi,si,ai], self.q[zi,gi,si,ai] = result[0], result[1], result[2],result[3], result[4]
                                    del result
                                else:
                                    self.psi[zi,gi,si,ai], self.ssq[zi,gi,si,ai], self.ssg[zi,gi,si,ai],self.sss[zi,gi,si,ai],self.ssa[zi,gi,si,ai], self.q[zi,gi,si,ai] =result[0], result[1], result[2], result[3],result[4], result[5]
                                    self.chi[zi,gi,si,ai] = self.params['alpha'] * self.psi[zi,gi,si,ai]
                                    del(result)
                            else:
                                self.psi[zi,gi,si,ai]=1
                                result = self.equations_region2(self.q[zi-1,gi,si,ai],self.ssq[zi-1,gi,si,ai],self.ssg[zi-1,gi,si,ai],self.sss[zi-1,gi,si,ai],self.ssa[zi,gi,si,ai],zi,gi,si,ai)
                                self.ssq[zi,gi,si,ai], self.ssg[zi,gi,si,ai],self.sss[zi,gi,si,ai],self.ssa[zi,gi,si,ai], self.q[zi,gi,si,ai] = result[0], result[1], result[2], result[3], result[4]
                                self.chi[zi,gi,si,ai] = np.maximum(self.z[zi],self.params['alpha'])
                                del result
            self.crisis_flag = np.zeros((self.Nz,self.Ng,self.Ns,self.Na))
            self.crisis_flag_bound = np.zeros((self.Nz,self.Ng,self.Ns,self.Na))
            for i in range(self.Ng):
                for j in range(self.Ns):
                    for k in range(self.Na):
                        if(self.crisis[i,j,k]<self.Nz-1):
                            self.ssq[int(self.crisis[i,j,k]),i,j,k] = self.ssq[int(self.crisis[i,j,k])-1,i,j,k]
                            self.ssg[int(self.crisis[i,j,k]),i,j,k] = self.ssg[int(self.crisis[i,j,k])-1,i,j,k]
                            self.ssa[int(self.crisis[i,j,k]),i,j,k] = self.ssa[int(self.crisis[i,j,k])-1,i,j,k]
            for i in range(self.Ng): 
                for j in range(self.Ns):
                    for k in range(self.Na):
                        self.crisis_flag[0:int(self.crisis[i,j,k]),i,j,k] = 1
            
            self.qz[1:self.Nz,:,:]  = (self.q [1:self.Nz,:,:,:] - self.q [0:self.Nz-1,:,:,:])/self.dz_mat; self.qz[0,:,:,:] = self.qz[1,:,:,:];
            self.qs[:,:,1:self.Ns,:]  = (self.q [:,:,1:self.Ns,:] - self.q [:,:,0:self.Ns-1,:])/self.ds_mat; self.qs[:,:,0,:] = self.qs[:,:,1,:];
            self.qg[:,1:self.Ng,:,:]  = (self.q [:,1:self.Ng,:,:] - self.q [:,0:self.Ng-1,:,:])/self.dg_mat; self.qg[:,0,:,:] = self.qg[:,1,:,:];
            self.qa[:,:,:,1:self.Na]  = (self.q [:,:,:,1:self.Na] - self.q [:,:,:,0:self.Na-1])/self.da_mat; self.qa[:,:,:,0] = self.qg[:,:,:,1];

            self.qzz[2:self.Nz,:,:,:] = (self.q[2:self.Nz,:,:,:] + self.q[0:self.Nz-2,:,:,:] - 2.*self.q[1:self.Nz-1,:,:,:])/(self.dz2_mat); self.qzz[0,:,:,:]=self.qzz[2,:,:,:]; self.qzz[1,:,:,:]=self.qzz[2,:,:,:]; 
            self.qss[:,:,2:self.Ns,:] = (self.q[:,:,2:self.Ns,:] + self.q[:,:,0:self.Ns-2,:] - 2.*self.q[:,:,1:self.Ns-1,:])/(self.ds2_mat); self.qss[:,:,0,:]=self.qss[:,:,2,:]; self.qss[:,:,1,:]=self.qss[:,:,2,:]
            self.qgg[:,2:self.Ng,:,:] = (self.q[:,2:self.Ng,:,:] + self.q[:,0:self.Ng-2,:,:] - 2.*self.q[:,1:self.Ng-1,:,:])/(self.dg2_mat); self.qgg[:,0,:,:]=self.qgg[:,2,:,:]; self.qgg[:,1,:,:]=self.qgg[:,2,:,:]
            self.qaa[:,:,:,2:self.Na] = (self.q[:,:,:,2:self.Na] + self.q[:,:,:,0:self.Na-2] - 2.*self.q[:,:,:,1:self.Na-1])/(self.da2_mat); self.qaa[:,:,:,0]=self.qaa[:,:,:,2]; self.qaa[:,:,:,1]=self.qaa[:,:,:,2]
            
            #need to fix this?
            q_temp = np.row_stack((self.q[0:1,:,:,:],self.q[:,:,:,:],self.q[self.q.shape[0]-1:self.q.shape[0],:,:,:]))
            q_temp = np.column_stack((q_temp[:,0:1,:,:],q_temp,q_temp[:,q_temp.shape[1]-1:q_temp.shape[1],:,:]))
            for gi in range(1,self.Ng):
                for zi in range(1,self.Nz):
                    self.Qzg[zi,gi,:,:]= (q_temp[zi+1,gi+1,:,:] - q_temp[zi+1,gi-1,:,:] - q_temp[zi-1,gi+1,:,:] + q_temp[zi-1,gi-1,:,:])/(4*self.dg_mat[zi-1,gi-1,:,:]*self.dz_mat[zi-1,gi-1,:,:]);
            del(q_temp)
            
            q_temp = np.row_stack((self.q[0:1,:,:,:],self.q[:,:,:,:],self.q[self.q.shape[0]-1:self.q.shape[0],:,:,:]))
            q_temp = np.dstack((q_temp[:,:,0:1,:],q_temp,q_temp[:,:,q_temp.shape[2]-1:q_temp.shape[2],:]))
            for si in range(1,self.Ns):
                for zi in range(1,self.Nz):
                            self.Qsz[zi,:,si]= (q_temp[zi+1,:,si+1,:] - q_temp[zi+1,:,si-1,:] - q_temp[zi-1,:,si+1,:] + q_temp[zi-1,:,si-1,:])/(4*self.ds_mat[zi-1,:,si-1,:]*self.dz_mat[zi-1,:,si-1,:]);
            del(q_temp)
            
            q_temp = np.row_stack((self.q[0:1,:,:,:],self.q[:,:,:,:],self.q[self.q.shape[0]-1:self.q.shape[0],:,:,:]))
            q_temp = np.concatenate((q_temp[:,:,:,0:1],q_temp,q_temp[:,:,:,q_temp.shape[3]-1:q_temp.shape[3]]),axis=3)
            for ai in range(1,self.Na):
                for zi in range(1,self.Nz):
                            self.Qza[zi,:,:,ai]= (q_temp[zi+1,:,:,ai+1] - q_temp[zi+1,:,:,ai-1] - q_temp[zi-1,:,:,ai+1] + q_temp[zi-1,:,:,ai-1])/(4*self.da_mat[zi-1,:,:,ai-1]*self.dz_mat[zi-1,:,:,ai-1]);
            del(q_temp)
            
            self.qzl  = self.qz/self.q; 
            self.qgl  = self.qg /self.q; 
            self.qsl  = self.qs /self.q; 
            self.qal  = self.qa /self.q; 
            self.qzzl = self.qzz/ self.q;
            self.qggl = self.qgg/self.q;
            self.qssl = self.qss/self.q;
            self.qaal = self.qaa/self.q;
            self.qgzl = self.Qzg/self.q;
            self.qszl = self.Qsz/self.q;
            self.qazl = self.Qza/self.q;
            
            self.iota = (self.q-1)/self.params['kappa']
            self.theta = self.chi/self.z_mat
            self.thetah = (1-self.chi)/(1-self.z_mat)
            self.theta[0,:,:,:] = self.theta[1,:,:,:]
            self.thetah[0,:,:,:] = self.thetah[1,:,:,:]
   
            self.consWealthRatioE = self.params['rho']
            self.consWealthRatioH = self.params['rho']
            self.sig_zk = self.z_mat*(self.theta-1)*self.ssq
            self.sig_zg = self.z_mat*(self.theta-1)*self.ssg
            self.sig_zs = self.z_mat *(self.theta-1)*self.sss
            self.sig_za = self.z_mat *(self.theta-1)*self.ssa
            self.sig_jk_e = self.dLogJe_z*self.sig_zk
            self.sig_jg_e = self.dLogJe_g*self.sig_g + self.dLogJe_z*self.sig_zg
            self.sig_js_e = self.dLogJe_s*self.sig_s + self.dLogJe_z*self.sig_zs
            self.sig_ja_e = self.dLogJe_a*self.sig_a + self.dLogJe_z*self.sig_za
            self.sig_jk_h = self.dLogJh_z*self.sig_zk
            self.sig_jg_h = self.dLogJh_g*self.sig_g + self.dLogJh_z*self.sig_zg
            self.sig_js_h = self.dLogJh_s*self.sig_s + self.dLogJh_z*self.sig_zs
            self.sig_ja_h = self.dLogJh_a*self.sig_a + self.dLogJh_z*self.sig_za
            if self.params['scale']>1:
                self.priceOfRiskE_k = -(1-self.params['gamma'])*self.sig_jk_e + self.sig_zk/self.z_mat + self.ssq + (self.params['gamma']-1)*self.params['sigma'] *np.sqrt(self.s_mat)
                self.priceOfRiskE_g = -(1-self.params['gamma'])*self.sig_jg_e + self.sig_zg/self.z_mat + self.ssg
                self.priceOfRiskE_s = -(1-self.params['gamma'])*self.sig_js_e + self.sig_zs/self.z_mat + self.sss
                self.priceOfRiskE_a = -(1-self.params['gamma'])*self.sig_ja_e + self.sig_za/self.z_mat + self.ssa
                self.priceOfRiskH_k = -(1-self.params['gamma'])*self.sig_jk_h - 1/(1-self.z_mat)*self.sig_zk + self.ssq + (self.params['gamma']-1)*self.params['sigma'] *np.sqrt(self.s_mat)
                self.priceOfRiskH_g = -(1-self.params['gamma'])*self.sig_jg_h - 1/(1-self.z_mat)*self.sig_zg + self.ssg
                self.priceOfRiskH_s = -(1-self.params['gamma'])*self.sig_js_h - 1/(1-self.z_mat)*self.sig_zs + self.sss
                self.priceOfRiskH_a = -(1-self.params['gamma'])*self.sig_ja_h - 1/(1-self.z_mat)*self.sig_za + self.ssa
            else:
                self.priceOfRiskE_k = -self.sig_jk_e + self.sig_zk/self.z_mat + self.ssq + (self.params['gamma']-1)*self.params['sigma'] *np.sqrt(self.s_mat)
                self.priceOfRiskE_g = -self.sig_jg_e + self.sig_zg/self.z_mat + self.ssg
                self.priceOfRiskE_s = -self.sig_js_e + self.sig_zs/self.z_mat + self.sss
                self.priceOfRiskE_a = -self.sig_ja_e + self.sig_za/self.z_mat + self.ssa
                self.priceOfRiskH_k = -self.sig_jk_h - 1/(1-self.z_mat)*self.sig_zk + self.ssq + (self.params['gamma']-1)*self.params['sigma'] *np.sqrt(self.s_mat)
                self.priceOfRiskH_g = -self.sig_jg_h - 1/(1-self.z_mat)*self.sig_zg + self.ssg
                self.priceOfRiskH_s = -self.sig_js_h - 1/(1-self.z_mat)*self.sig_zs + self.sss
                self.priceOfRiskH_a = -self.sig_ja_h - 1/(1-self.z_mat)*self.sig_za + self.ssa
            self.rp = self.priceOfRiskE_k*self.ssq + self.priceOfRiskE_g*self.ssg + self.priceOfRiskE_s*self.sss + self.priceOfRiskE_a*self.ssa
            self.rp_ = self.priceOfRiskH_k*self.ssq + self.priceOfRiskH_g*self.ssg + self.priceOfRiskH_s*self.sss + self.priceOfRiskH_a * self.ssa
            
            self.mu_z = self.z_mat*((self.a_mat - self.iota)/self.q + self.g_mat - self.consWealthRatioE + (self.theta-1)*(self.ssq*(self.priceOfRiskE_k - self.ssq) + self.ssg*(self.priceOfRiskE_g - self.ssg) +self.sss*(self.priceOfRiskE_s - self.sss) +self.ssa*(self.priceOfRiskE_a - self.ssa) ) + 
                                    (1-self.params['alpha'])*(self.ssq*(self.priceOfRiskE_k - self.priceOfRiskH_k) + self.ssg*(self.priceOfRiskE_g - self.priceOfRiskH_g)+ self.sss*(self.priceOfRiskE_s - self.priceOfRiskH_s) + self.ssa*(self.priceOfRiskE_a - self.priceOfRiskH_a))) - self.params['hazard_rate1']*self.z_mat - self.crisis_flag*self.params['hazard_rate2'] * self.z_mat
            self.mu_g = self.params['lambda_g']*(self.params['g_avg'] - self.g_mat)
            self.mu_s = self.params['lambda_s']*(self.params['s_avg'] - self.s_mat)
            self.mu_a = self.params['lambda_a']*(self.params['a_avg'] - self.a_mat)
            self.growthRate = np.log(self.q)/self.params['kappa'] -self.params['delta'] + self.g_mat
            self.Phi = np.log(self.q)/self.params['kappa']
            self.mu_q = self.qzl*self.mu_z + self.qgl*self.mu_g + self.qsl*self.mu_s + self.qsl*self.mu_s +self.qal*self.mu_a+ 0.5*self.qzzl*(self.sig_zk**2 + self.sig_zg**2 + self.sig_zs**2 + self.sig_za**2) +\
                    0.5*self.qggl*self.sig_g**2 + 0.5*self.qssl*self.sig_s**2 + 0.5*self.qaal*self.sig_a**2+ self.qgzl*(self.sig_zg * self.sig_g) + self.qszl*(self.sig_zg*self.sig_s) + self.qazl*(self.sig_za*self.sig_a)
            self.r = self.crisis_flag*(-self.rp_ + (self.params['aH'] - self.iota)/self.q + self.Phi + self.g_mat - self.params['delta'] + self.mu_q + self.params['sigma'] * np.sqrt(self.s_mat)*(self.ssq - np.sqrt(self.s_mat) * self.params['sigma'])) +\
                    (1-self.crisis_flag)*(-self.rp + (self.a_mat - self.iota)/self.q + self.Phi + self.g_mat - self.params['delta'] + self.mu_q + self.params['sigma'] * np.sqrt(self.s_mat)* (self.ssq - np.sqrt(self.s_mat) * self.params['sigma']))
            for gi in range(self.Ng):
                for si in range(self.Ns):
                    for ai in range(self.Na):
                        crisis_temp = np.where(self.crisis_flag[:,gi,si,ai]==1.0)[0][-1]+1
                        try:
                            self.r[crisis_temp-1:crisis_temp+2,gi,si,ai] = 0.5*(self.r[crisis_temp+3,gi,si,ai] + self.r[crisis_temp-2,gi,si,ai]) #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
                        except:
                            print('no crisis')
                
            self.diffusion_z = 0.5*(self.sig_zk**2 + self.sig_zg**2 + self.sig_zs**2)
            self.diffusion_g = 0.5*(self.sig_g)**2
            self.diffusion_s = 0.5*(self.sig_s)**2
            self.diffusion_a = 0.5*(self.sig_a)**2
            self.cross_term_zg = self.sig_zg*self.sig_g
            self.cross_term_zs = self.sig_zs*self.sig_s
            self.cross_term_za = self.sig_za*self.sig_a
            
            if self.params['scale']>1:
                self.advection_z_e = self.mu_z
                self.advection_z_h = self.mu_z
                self.linearTermE = -(0.5*self.params['gamma']*(self.sig_jk_e**2 + self.sig_jg_e**2 + self.sig_js_e**2 + self.sig_ja_e**2 +self.params['sigma']**2*self.s_mat) -\
                                    self.growthRate + (self.params['gamma']-1)*(self.sig_jk_e*self.params['sigma'] *np.sqrt(self.s_mat)) -\
                                    self.params['rho']*(np.log(self.params['rho']) - np.log(self.Je) + np.log(self.z_mat*self.q))) + (self.params['hazard_rate1'] + self.crisis_flag*self.params['hazard_rate2'])*((self.Jh/self.Je)**(1-self.params['gamma'])-1)/(1-self.params['gamma']) 
                self.linearTermH = -(0.5*self.params['gamma']*(self.sig_jk_h**2 + self.sig_jg_h**2 + self.sig_js_h**2 + self.sig_ja_h**2+self.params['sigma']**2 * self.s_mat) -\
                                    self.growthRate + (self.params['gamma']-1)*(self.sig_jk_h*self.params['sigma'] * np.sqrt(self.s_mat)) -\
                                    self.params['rho']*(np.log(self.params['rho']) - np.log(self.Jh) + np.log((1-self.z_mat)*self.q)))
            else:
                self.advection_z_e = self.mu_z + (1-self.params['gamma'])*(self.params['sigma'] * np.sqrt(self.s_mat) * self.sig_zk)
                self.advection_z_h = self.mu_z + (1-self.params['gamma'])*(self.params['sigma'] * np.sqrt(self.s_mat) * self.sig_zk)
                self.linearTermE = (1-self.params['gamma'])*(self.growthRate - 0.5*self.params['gamma']*self.params['sigma']**2 * self.s_mat +\
                              self.params['rho']*(np.log(self.params['rho']) + np.log(self.q*self.z_mat))) -  self.params['rho']* np.log(self.Je)
                self.linearTermH = (1-self.params['gamma'])*(self.growthRate - 0.5*self.params['gamma']*self.params['sigma']**2 * self.s_mat  +\
                              self.params['rho']*(np.log(self.params['rho']) + np.log(self.q*(1-self.z_mat)))) -   self.params['rho'] * np.log(self.Jh)
            
            #Time step
            #data prep
            if pde=='True':
                if self.amax < 0.01: 
                    learning_rate = 0.001
                    layers = [5, 30, 30, 30, 30, 1]
                    self.dt = 0.5
                else:
                    learning_rate = 0.001
                    layers = [5, 30, 30, 30, 30, 1]
                    self.dt = 0.5
                tb = np.vstack((0,self.dt)).astype(np.float32)
                
                #need a better way to broadcast
                aa,ss,zz = np.meshgrid(self.a,self.s,self.z)
                X_temp = np.vstack((np.ravel(zz),np.ravel(ss),np.ravel(aa),np.full(np.ravel(aa).shape[0],self.dt))).T
                g_tile = np.repeat(self.g,np.ravel(zz).shape[0])
                X_temp = np.tile(X_temp.transpose(),self.Ng).T
                X = np.insert(X_temp,1,g_tile,axis=1)
                del X_temp
                X_f = np.vstack((X[:,0:-1].T,np.random.uniform(0,self.dt,X.shape[0]).reshape(-1))).T
                x_star = np.vstack((X[:,0:-1].T,np.full(X.shape[0],0).reshape(-1))).T
                def flatten(obj):
                    return obj.transpose().reshape(-1)
                Jhat_e0 = self.Je.copy().transpose().flatten().reshape(-1,1)
                Jhat_h0 = self.Jh.copy().transpose().flatten().reshape(-1,1)
                
                #active learning
                crisis_min,crisis_max = int(self.crisis.min()), int(self.crisis.max())
                X_,X_f_ = X.copy(),X_f.copy()
                X_f_plot = X_f.copy()
                def flatten(obj):
                    return obj.transpose().reshape(-1)
                Jhat_e0_,Jhat_h0_ = Jhat_e0.copy(), Jhat_h0.copy()

                def add_crisis_points(vector):
                    new_vector = vector.copy()
                    shift = 0
                    for i in range(self.Na):
                        new_vector = np.vstack((new_vector,vector[crisis_min-20+shift+ i*self.Nz : crisis_max+20+shift+i*self.Nz,:]))
                    shift = self.Na*self.Nz
                    for j in range(self.Ns):
                        new_vector = np.vstack((new_vector,vector[crisis_min-20+shift+j*self.Nz: crisis_max+20+shift+j*self.Nz,:]))
                    shift += self.Ns*self.Nz
                    for k in range(self.Ng):
                        new_vector = np.vstack((new_vector,vector[crisis_min-20+shift+j*self.Nz: crisis_max+20+shift+j*self.Nz,:]))
                    return new_vector
                
                def sample_boundary_points1():
                    boundary_points = []
                    for i in range(self.Ng):
                        boundary_points.append(np.arange(i*self.Nz,(i*self.Nz + 50)))
                    return boundary_points
                def sample_boundary_points2():
                    boundary_points = []
                    for i in range(1,self.Ng):
                        boundary_points.append(np.arange(i*self.Nz -50,(i*self.Nz + 1)))
                    return boundary_points
                boundary_points1= np.array(sample_boundary_points1()).flatten()
                boundary_points2= np.array(sample_boundary_points2()).flatten()
                X_,X_f_,Jhat_e0_,Jhat_h0_ = add_crisis_points(X),add_crisis_points(X_f),add_crisis_points(Jhat_e0),add_crisis_points(Jhat_h0)
                diffusion_z, diffusion_g, diffusion_s, diffusion_a = add_crisis_points(self.diffusion_z.transpose().reshape(-1,1)), add_crisis_points(self.diffusion_g.transpose().reshape(-1,1)), add_crisis_points(self.diffusion_s.transpose().reshape(-1,1)), add_crisis_points(self.diffusion_a.transpose().reshape(-1,1))
                advection_z_e, advection_z_h, advection_g, advection_s, advection_a = add_crisis_points(self.advection_z_e.transpose().reshape(-1,1)), add_crisis_points(self.advection_z_h.transpose().reshape(-1,1)), add_crisis_points(self.mu_g.transpose().reshape(-1,1)), \
                                                                                        add_crisis_points(self.mu_s.transpose().reshape(-1,1)), add_crisis_points(self.mu_a.transpose().reshape(-1,1))
                                                    
                cross_term_zg, cross_term_zs, cross_term_za, linearTermE, linearTermH = add_crisis_points(self.cross_term_zg.transpose().reshape(-1,1)), add_crisis_points(self.cross_term_zs.transpose().reshape(-1,1)), add_crisis_points(self.cross_term_za.transpose().reshape(-1,1)),\
                                                                                        add_crisis_points(self.linearTermE.transpose().reshape(-1,1)), add_crisis_points(self.linearTermH.transpose().reshape(-1,1))
                
                crisisPointsLength = X_.shape[0]-X.shape[0]
                if self.params['active']=='on':
                    idx1 = np.random.choice(X_.shape[0],100000,replace=False)
                    idx2 = np.random.choice(np.arange(X_.shape[0]-crisisPointsLength,X_.shape[0]),2000,replace=True)
                    idx = np.hstack((idx1,idx2))
                else:
                    idx = np.random.choice(X_.shape[0],10000,replace=False)
                
                X_, X_f_, Jhat_e0_, Jhat_h0_ = X_[idx], X_f_[idx], Jhat_e0_[idx], Jhat_h0_[idx]
                diffusion_z_tile = diffusion_z.transpose().reshape(-1)[idx]
                diffusion_g_tile = diffusion_g.transpose().reshape(-1)[idx]
                diffusion_s_tile = diffusion_s.transpose().reshape(-1)[idx]
                diffusion_a_tile = diffusion_a.transpose().reshape(-1)[idx]
                advection_z_e_tile = advection_z_e.transpose().reshape(-1)[idx]
                advection_g_tile = advection_g.transpose().reshape(-1)[idx]
                advection_s_tile = advection_s.transpose().reshape(-1)[idx]
                advection_a_tile = advection_a.transpose().reshape(-1)[idx]
                advection_z_h_tile = advection_z_h.transpose().reshape(-1)[idx]
                cross_term_zg_tile = cross_term_zg.transpose().reshape(-1)[idx]
                cross_term_zs_tile = cross_term_zs.transpose().reshape(-1)[idx]
                cross_term_za_tile = cross_term_za.transpose().reshape(-1)[idx]
                linearTermE_tile = linearTermE.transpose().reshape(-1)[idx]
                linearTermH_tile = linearTermH.transpose().reshape(-1)[idx]
                
                #sovle the PDE
                model_E = nnpde_informed(-linearTermE_tile.reshape(-1,1), advection_z_e_tile.reshape(-1,1),advection_g_tile.reshape(-1,1),advection_s_tile.reshape(-1,1),advection_a_tile.reshape(-1,1), 
                                         diffusion_z_tile.reshape(-1,1),diffusion_g_tile.reshape(-1,1),diffusion_s_tile.reshape(-1,1),diffusion_a_tile.reshape(-1,1),
                                         cross_term_zg_tile.reshape(-1,1),cross_term_zs_tile.reshape(-1,1),cross_term_za_tile.reshape(-1,1), Jhat_e0_.reshape(-1,1).astype(np.float32),X_,layers,X_f_,self.dt,tb,learning_rate,self.params['maxEpochs_e'])
                model_E.train()
                newJeraw = model_E.predict(x_star)
                model_E.sess.close()
                newJe = np.swapaxes(newJeraw.transpose().reshape(self.Ng,self.Ns,self.Na,self.Nz).transpose(),1,3)
               
                del model_E
                
                model_H = nnpde_informed(-linearTermH_tile.reshape(-1,1), advection_z_h_tile.reshape(-1,1),advection_g_tile.reshape(-1,1),advection_s_tile.reshape(-1,1),advection_a_tile.reshape(-1,1), 
                                         diffusion_z_tile.reshape(-1,1),diffusion_g_tile.reshape(-1,1),diffusion_s_tile.reshape(-1,1),diffusion_a_tile.reshape(-1,1),
                                         cross_term_zg_tile.reshape(-1,1),cross_term_zs_tile.reshape(-1,1),cross_term_za_tile.reshape(-1,1), Jhat_h0_.reshape(-1,1).astype(np.float32),X_,layers,X_f_,self.dt,tb,learning_rate,self.params['maxEpochs_h'])
                model_H.train()
                newJhraw = model_H.predict(x_star)
                model_H.sess.close()
                newJh = np.swapaxes(newJhraw.transpose().reshape(self.Ng,self.Ns,self.Na,self.Nz).transpose(),1,3)
    
                self.ChangeJe = np.abs(newJe - self.Je)
                self.ChangeJh = np.abs(newJh - self.Jh)
                if self.params['scale']>1: cutoff = 1
                else: cutoff = 10
                self.relChangeJe = np.abs((newJe[cutoff:-cutoff,:,:,:] - self.Je[cutoff:-cutoff,:,:,:]) / (self.Je[cutoff:-cutoff,:,:,:]))
                self.relChangeJh = np.abs((newJh[cutoff:-cutoff,:,:,:] - self.Jh[cutoff:-cutoff,:,:,:]) / (self.Jh[cutoff:-cutoff,:,:,:]))
                
                self.Jh = newJh
                self.Je = newJe
                
                if np.sum(np.isnan(newJe))>0 or np.sum(np.isnan(newJh))>0:
                    print('NaN values found in Value function')
                    break
                if self.params['scale']>1:
                    self.amax = np.maximum(np.amax(self.ChangeJe),np.amax(self.ChangeJh))
                else:
                    self.amax = np.maximum(np.amax(self.relChangeJe),np.amax(self.relChangeJh))
                del model_H, X, X_f, x_star, tb
                self.amax_vec.append(self.amax)
                #memory management
                del self.diffusion_a,self.diffusion_s,self.diffusion_z,self.diffusion_g,self.cross_term_za,self.cross_term_zs,self.cross_term_zg
                np.save('4Dmodel_Je', self.Je)
                np.save('4Dmodel_Jh', self.Jh)
                np.save('4Dmodel_error',self.amax_vec)
                if self.params['save']=='yes': self.pickle_stuff(self,'model4D' + '.pkl') 
                
                def plot_grid(data,name):
                    mypoints = []
                    mypoints.append([data[:,0],data[:,1],data[:,2]])
                    
                    data = list(zip(*mypoints))           # use list(zip(*mypoints)) with py3k  
                    
                    fig = pyplot.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(data[0], data[1],data[2])
                    ax.set_xlabel('\n' +'Wealth Share (z)',fontsize=15)
                    ax.set_ylabel('\n' +'Productivity (a)',fontsize=15)
                    ax.set_zlabel('\n' +'Time (t)')
                    ax.set_xlim3d(0,1)
                    ax.set_zlim3d(0,self.dt)
                    ax.set_ylim3d(self.f_l,self.f_u)
                    ax.set_title(name,fontsize=20)
                    pyplot.show()
                    fig.savefig(plot_path + str(name) +'.png')
                if False:  #self.Iter == 1:
                    plt.style.use('classic')
                    if not os.path.exists('../output/plots/extended/'):
                        os.mkdir('../output/plots/extended/')
                    plot_path = '../output/plots/extended/'
                    plot_grid(X_f_plot,'Full grid')
                    plot_grid(X_f_,'Training sample')
                    plt.style.use('seaborn')
                if self.amax < self.convergenceCriterion:
                    self.converged = 'True'
                    break
                print('Absolute max of relative error: ',self.amax)
            else:
                self.A = self.psi*(self.a_mat) + (1-self.psi) * (self.params['aH'])
                self.AminusIota = self.psi*(self.a_mat - self.iota) + (1-self.psi) * (self.params['aH'] - self.iota)
                self.pd = np.log(self.q / self.AminusIota)
                break
        
            print('Iteration no: ',self.Iter )
        if self.converged == 'True':
            print('Algortihm converged after {} time steps.\n'.format(timeStep));
        else:
            print('Algorithm terminated without convergence after {} time steps.'.format(timeStep));
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
if __name__ =="__main__":
    params={'rho': 0.05, 'aH': 0.02,
            'alpha':1.0, 'kappa':5, 'delta':0.05, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.15, 'gamma':5,
             'lambda_s' : 0.5,'lambda_g':0.5,'lambda_a':0.5, 'g_u' : 0.05, 'g_l' : -0.05, 'g_avg': 0.0,
             's_u' : 0.2, 's_l' : 0.1, 's_avg': 0.15, 'beta_g':12,'beta_s':12,
             'a_u' : 0.2, 'a_l' : 0.1, 'a_avg': 0.15, 'beta_a':8,'hazard_rate1' :0.065,'hazard_rate2':0.45};
    params['scale']=2
    params['load'] = args.load
    params['save'] = args.save
    params['active'] = args.active
    params['maxEpochs_e'] = args.epochsExperts
    params['maxEpochs_h'] = args.epochsHouseholds
    ext = model4D(params)
    ext.maxIterations=50
    ext.solve(pde='True')
    