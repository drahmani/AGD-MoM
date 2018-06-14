def gmm_gen(Mu,w,Sigma,N):
    [K,D] = Mu.shape
    z=np.zeros([N,D])
    x=np.zeros([N,D])
    log_i=np.zeros([N])
    for i in range(N):
        ind = np.where(np.cumsum(w)> np.random.uniform())
        indy = ind[0][0]
        log_i[i] = indy
        z[i,:]= np.random.multivariate_normal(np.zeros([D]),np.squeeze(Sigma[:,:,indy]))
        x[i,:]=Mu[indy,:]+z[i,:];
    return x,log_i;

def gen_mom(x,K):
#if True:    
    [N,D]=x.shape
    cov_x=np.cov(np.transpose(x))
    me_x=np.mean(x,axis=0)
    eig_x=LA.eig(cov_x)
    sigma=sorted(eig_x[0])[0]
    idx=np.argsort(eig_x[0])[0]
    e_vec=eig_x[1][:,idx] 
    M_w=np.mean(x,axis=0)
    M1=np.dot(x.T,np.power(np.dot(e_vec,((x-M_w).T)),2))/N#not sure it is true!!
#empirical second moment
    M=np.dot(x.T,x)/N-sigma*np.eye(D)
#empirical third order moment
    e=np.eye(D)
    T_l=np.zeros([D,D*D])
    T_x=np.zeros([D,D,D])
    for i in range(D):
       T_l= T_l+np.kron(np.outer(M1.T,e[i,:]),e[i,:])+np.kron(np.outer(e[i,:].T,M1),e[i,:])+np.kron(np.outer(e[i,:].T,e[i,:]),M1)
    T_l=np.reshape(T_l,[D,D,D],0)
    for t in range(N):
      for i in range(D):
         for j in range(D):
            for l in range(D):
                T_x[i,j,l]=T_x[i,j,l]+x[t,i]*x[t,j]*x[t,l];
    T=T_x/N-T_l;
    return T,M,sigma;

if True:
    import numpy as np
    import tensorflow as tf
    from tensorflow.python import debug as tf_debug
    from datetime import datetime
    import matplotlib.pyplot as plt
    from numpy import linalg as LA
    import importlib as imp
    import sys
    import gen_gmm as gmm
    import scipy.io
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    import pickle
    stem = '/Users/donya/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents -Donya’s Mac/tfcode'
    import os
    if not os.path.isdir(stem):
        os.mkdir(stem)

# Load the data.
if True:
    Mup = np.array([[1,2],[5,9]])
    wp = np.array([.5,.5])
    K=wp.size
    Sigma=np.zeros([Mup.shape[1],Mup.shape[1],Mup.shape[0]])
    Sigma[:,:,0] = np.diag([1,1])*1
    Sigma[:,:,1] = np.diag([1,1])*1
    N=500
    makey = True
    if makey==True:
        x,log_i = gmm.gmm_gen(Mup,wp,Sigma,N)
        np.savetxt('x.txt',x)
        np.savetxt('log_i.txt',log_i)
    else:
        x = np.loadtxt('x.txt')
        log_i = np.loadtxt('log_i.txt')

    Tin,Min,sigma=gmm.gen_mom(x,K)
    nblks = 1

 # Ok so now we have the inputs, T and M from data x
# Define the placeholder inputs.
    tf.reset_default_graph() # Clear the graph
    T = tf.placeholder(tf.float32, shape=(None,Tin.shape[0],Tin.shape[1],Tin.shape[2]),name = "T")
    M = tf.placeholder(tf.float32, shape=(None,Min.shape[0],Min.shape[1]),name = "M")
    
# Define the variables and their initialisations
if True:    
  with tf.name_scope("Init"):
    K=2
    D=2
    init = tf.abs(tf.truncated_normal((K,D),stddev = 2))
    Testy = False
    if Testy == False: # False: random choice True: set up a value
        Mu=tf.Variable(init,name="Mu")
        w = tf.Variable(tf.random_uniform((K,)),name='w')
    else: 
        Mu= tf.Variable([[.5,2.5],[3.3,6.3]],dtype = tf.float32,name="Mu")
        w = tf.Variable([.3 ,.6],dtype = tf.float32,name='w')
    Lambda = tf.constant([10],dtype = tf.float32, name = 'Lambda')

# Batch feeding function
def fetch_batch(epoch,batch_index,batch_size):
        Tb = np.zeros([1,D,D,D])
        Mb = np.zeros([1,D,D])
        Mb[0,:,:] = Min
        Tb[0,:,:,:] = Tin
        return Tb,Mb

# Define the graph.
T_hat=np.zeros([D,D,D])
M_hat=np.zeros([D,D])
with tf.name_scope("MOM"):
   for i in range(K):
      MuMu0  = tf.einsum('m,n->mn',Mu[i,:],Mu[i,:])
      with tf.name_scope("T_est"):
      #if True:
        for j in range(K):
           MuMuMu0 = tf.einsum('mn,l->mnl',MuMu0,Mu[j,:])       
           wn=tf.abs(w)/tf.reduce_sum(tf.abs(w))
           wMuMuMu0 = tf.scalar_mul(wn[j],MuMuMu0)    
           T_hat += wMuMuMu0
      with tf.name_scope("M_est"):
        for j in range(K):
           wMuMu0 = tf.scalar_mul(wn[j],MuMu0)   
           M_hat += wMuMu0#to see results M_hat.eval()        
   mse = tf.reduce_mean(tf.square(T_hat-T),name = 'mse')
   mse_summary =tf.summary.scalar('WMSE',mse) # 

# Loss function
with tf.name_scope("loss"):
    error = T-T_hat
    L1 = tf.norm(error)
    error2 = M-M_hat
    L2 = tf.norm(error2)
    lL2 = L2*Lambda
    loss = tf.add(L1,lL2)

# Training function

with tf.name_scope("train"):
    optimiser1 = tf.train.MomentumOptimizer(1e-4,.95)#GradientDescentOptimizer,AdadeltaOptimizer
    optimiser2 = tf.train.MomentumOptimizer(1e-5,.95)
    training_op_mu = optimiser1.minimize(loss,var_list = [Mu])
    training_op_w = optimiser2.minimize(loss,var_list = [w])
with tf.name_scope("eval"):
   mse = loss



# Set up the session
n_epochs = 1000
batch_size = 1
n_batches = nblks

# Training Session
#debug_flag =False
#if debug_flag:
mse_log=np.zeros([n_epochs])
Mu1_all=np.zeros([n_epochs,D])
Mu2_all=np.zeros([n_epochs,D])
w_all=np.zeros([n_epochs,K])
sess = tf.InteractiveSession() 
# for iter in range(100):
# Initialisation
init = tf.global_variables_initializer()
sess.run(init)

if True:
    Tb = np.zeros([1,D,D,D])
    Mb = np.zeros([1,D,D])
    Mb[0,:,:] = Min
    Tb[0,:,:,:] = Tin
    sum_squares=np.zeros([N,K])
# Try giving it the answer and see if it stays there
at_best = mse.eval(feed_dict = {T: Tb, M: Mb, w:wp,Mu:Mup})[0]
for epoch in range(n_epochs):
    for batch_index in range(n_batches):
        Tb,Mb = fetch_batch(epoch,batch_index,batch_size)
        for i in range(1):
           sess.run(training_op_mu,feed_dict = {T: Tb, M: Mb})
           sess.run(training_op_w,feed_dict = {T: Tb, M: Mb})
    summary_str = mse_summary.eval(feed_dict = {T: Tb, M: Mb})

    acc_test = loss.eval(feed_dict = {T: Tb, M: Mb})
    Mu_test = Mu.eval()
    w_test = wn.eval()
    Tb = np.zeros([1,D,D,D])
    Mb = np.zeros([1,D,D])
    Mb[0,:,:] = Min
    Tb[0,:,:,:] = Tin
    acc_train = mse.eval(feed_dict = {T: Tb, M: Mb})
    mse_log[epoch]=acc_train[0]
    Mu1_all[epoch,:]=Mu_test[0]
    Mu2_all[epoch,:]=Mu_test[1]
    w_all[epoch,:]=w_test

history = [mse_log,Mu1_all,Mu2_all,w_all]
with open(stem + 'history_' + str(index)+'.txt','wb') as fp:
    pickle.dump(history,fp)
