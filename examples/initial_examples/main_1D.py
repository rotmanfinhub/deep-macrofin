

"""
Author: Goutham G.
Description: One capital two agent model. Only state variable is eta (e in the code - that is wealth share of speculators).
Status: Solved using Pytorch.
"""
from pyDOE import lhs
import torch
import numpy as np 
#torch.manual_seed(123)
#np.random.seed(123)
import torch.optim as optim
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
#import matplotlib as mpl
#mpl.use('TkAgg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import glob
import os
# files = glob.glob('./plot*')
# for f in files:
#     os.remove(f)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import json
import random

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## Economic Parameters
# Save everything in a dictionary, for clarity
params = {
    "gammai": 1.0,
    "gammah": 2.0,
    "rhoi": 0.05,
    "rhoh": 0.05,
    "siga" : 0.2, #\sigma^{a}
    "mua":0.04,
    "muO":0.04,
    "n_pop": 2,
    "zetai":1.00005,
    "zetah":1.00005,
    "aa":0.1,
    "batchSize":100,
    "nn_width":30,
    "nn_num_layers":4,
    "neta":100,
    "start_eta":0.01,
    "end_eta":0.99,
    "kappa":10000
}

class Net1(torch.nn.Module):
    def __init__(self, nn_width, nn_num_layers,positive=False):
        super(Net1, self).__init__()
        layers = [torch.nn.Linear(1, nn_width), torch.nn.Tanh()]
        for i in range(1, nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(nn_width, 1))
        self.positive = positive
        self.net = torch.nn.Sequential(*layers)
        torch.nn.init.xavier_normal_(self.net[0].weight)  # Initialize the first linear layer weights

    def forward(self, X):
        output = self.net(X)
        if self.positive: output = torch.nn.functional.softplus(output)  # Apply softplus to the output
        return output

class Training_Sampler():
    def __init__(self, para_dict):
        self.params = para_dict

    def sample(self,N):
        ''' Construct share by renormalization
        '''
        # a = lhs(self.params['n_pop'], N) +0.02
        # eta = a/np.sum(a, axis=1, keepdims=True)*np.ones((1,params['n_pop'])) # renormalization
        # return eta[:,:-1]
        return np.random.uniform(0., 1., (N, 1))

class Training_pde():

    def __init__(self, para_dict):
        self.params=para_dict
    
    def get_derivs_1order(self,y,x):
        """ Returns the first order derivatives,
            Automatic differentiation used
        """
        dy_dx = torch.autograd.grad(y, x,
                            create_graph=True,
                            grad_outputs=torch.ones_like(y))[0]
        return dy_dx ## Return 'automatic' gradient.

    def loss_fun(self,Xi,Xh,Mue,Sigea,Qa,Wia,Wha,X):
        e = X.clone() # state variable
        e.requires_grad_(True)
        
        # agent action + endogenous variables
        xi   = Xi(e) # agent i (\xi_t^i)
        xh   = Xh(e) # agent h (\xi_t^h)
        mue = Mue(e) # mu_t^{\eta}
        sigea = Sigea(e) # \sigma_t^{\eta a}
        qa = Qa(e) # price of the capital
        wia = Wia(e) # w in budget constraint for agent i
        wha = Wha(e) # w in budget constraint for agent h

        # New variable definitions using equations
        iota_a = (qa-1)/self.params['kappa'] # iota in Phi(iota_a) before dt in the capital dynamics
        Phi_a = 1/self.params['kappa']*torch.log(1+self.params['kappa']*iota_a) # this is how Phi(iota_a) is computed?
        
        ci = (self.params['rhoi']**self.params['zetai'])*xi**(1 - self.params['zetai']) # ci and ch in market clearing conditions 
        ch = (self.params['rhoh']**self.params['zetah'])*xh**(1 - self.params['zetah'])

        
        # automatic differentiations for qa
        qa_e   = self.get_derivs_1order(qa,e)[:,0].unsqueeze(1) 
        qa_ee    = self.get_derivs_1order(qa_e,e)[:,0].unsqueeze(1) 

        # automatic differentiations for xie and xih
        xi_e  = self.get_derivs_1order(xi,e)[:,0].unsqueeze(1) 
        xi_ee  = self.get_derivs_1order(xi_e,e)[:,0].unsqueeze(1) 
        
        xh_e  = self.get_derivs_1order(xh,e)[:,0].unsqueeze(1) 
        xh_ee  = self.get_derivs_1order(xh_e,e)[:,0].unsqueeze(1) 
        
        sigqaa = qa_e/qa*sigea*e # eq 21
        signia = wia*(self.params['siga']+sigqaa) # sigma in budget constraint dymanic
        signha = wha*(self.params['siga']+sigqaa)
        sigxia = xi_e/xi*sigea*e # eq 22
        sigxha = xh_e/xh*sigea*e # eq 23
        signa = e*signia + (1-e)*signha # definition line 36

        muqa = qa_e/qa*mue*e + 1/2*qa_ee/qa*( sigea**2 )*e**2  # eq 24
        rka  = (self.params['aa'] - iota_a)/qa + self.params['mua'] + Phi_a + muqa + sigqaa*self.params['siga'] # eq 3, the bracket before dt, missing iota_t^a in the first term? used for w^{ha}
        r = rka  - self.params['gammah']*wha*((self.params['siga'] + sigqaa) )**2\
                       + (1-self.params['gammah'])*sigxha*((self.params['siga'] + sigqaa)) # eq 15
        muni = r - ci + wia*(rka- r) # \mu^{nj} in eq 5
        munh = r - ch + wha*(rka-r) 
        muqk = (muqa + self.params['mua'] + Phi_a + self.params['siga']*sigqaa)  # not found and not used
        
        muxi = (xi_e * mue *e  + 0.5*(xi_ee*(sigea**2 )*e**2 ))/xi # eq 25
        muxh = (xh_e * mue *e  + 0.5*(xh_ee*(sigea**2 )*e**2 ))/xh # eq 26
        # the following lineari and linearh are the rewritten HJB equations (eq 33) missing muxi and muhi respectively
        lineari = self.params['rhoi']/(1-1/self.params['zetai']) * ((ci/xi)**(1-1/self.params['zetai'])-1) + muxi + muni-\
                    self.params['gammai']/2*(signia**2) - self.params['gammai']/2*(sigxia**2) +\
                    (1-self.params['gammai'])* (sigxia*signia)
                    
        linearh = self.params['rhoh']/(1-1/self.params['zetah']) * ((ch/xh)**(1-1/self.params['zetah'])-1) + muxh + munh-\
                    self.params['gammah']/2*(signha**2) - self.params['gammah']/2*(sigxha**2) +\
                    (1-self.params['gammah'])* (sigxha*signha)
                     
                     
        rka_hat = rka + (self.params['muO']-self.params['mua'])/self.params['siga']*(self.params['siga']+sigqaa) # rka after noise added eq 3, used for w^{ia}
        yy = r + wia**2*(self.params['siga']+sigqaa)**2 - wia * (self.params['muO']-self.params['mua'])/self.params['siga'] *(self.params['siga']+sigqaa) + (self.params['siga']+sigqaa)**2*(1-wia) - ci - muqk
        #xx = r+ wia*(rka-r)-ci-muqk
        #solve for inner loop

        # endogenous equations, setting constraints on the problems.
        # We want to make sure the endogenous to always be satisfied
        # why do we need the zeros?
        eq1 = torch.zeros_like(e) 
        eq2 = (1-e)*(muni-munh) + signa**2 - signia*signa - mue # eq 27
        #eq2 = xx - mue
        eq3 = sigea - (1-e)*(signia - signha) # eq 28
        eq4 = torch.zeros_like(e) 
        eq5 = torch.zeros_like(e) 
        eq6 = (rka_hat-r) -self.params['gammai']*wia*  ((self.params['siga'] + sigqaa))**2 + (1-self.params['gammai'])*sigxia*(self.params['siga'] + sigqaa) # eq 29
        # eq6 = torch.zeros_like(e)
        # eq7 = (rka-r) -self.params['gammah']*wha*  ((self.params['siga'] + sigqaa))**2 - (1-self.params['gammah'])*sigxha*(self.params['siga'] + sigqaa) # eq 30
        eq7 = torch.zeros_like(e)
        eq8 = wia*e + wha*(1-e) - 1 # eq 31
        eq9 = torch.zeros_like(e) 
        eq10 = (ci*e + ch*(1-e))*qa - (self.params['aa'] -iota_a) # eq 32
        eq11 = torch.zeros_like(e) 
    
        hjb_i = (lineari)

        hjb_h = (linearh)
        
        return eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8,eq9,eq10,eq11,hjb_i,hjb_h

nn_width = params['nn_width']
nn_num_layers = params['nn_num_layers']

set_seeds(0)
TP  = Training_pde(params)
TS  = Training_Sampler(params)
Xi      = Net1(nn_width,nn_num_layers,True).to(device) 
Xh      = Net1(nn_width,nn_num_layers,True).to(device) 
Mue       = Net1(nn_width,nn_num_layers).to(device) #\mu_t^{\eta}
Qa      = Net1(nn_width,nn_num_layers,True).to(device) #q_t^a
Wia      = Net1(nn_width,nn_num_layers).to(device) #w_t^{ia}
Wha      = Net1(nn_width,nn_num_layers).to(device) #w_t^{ha}
Sigea       = Net1(nn_width,nn_num_layers).to(device) #\sigma_t^{\eta a}


Best_model_Xi      = Net1(nn_width,nn_num_layers,True).to(device) 
Best_model_Xh       = Net1(nn_width,nn_num_layers,True).to(device) 
Best_model_Mue       = Net1(nn_width,nn_num_layers).to(device)
Best_model_Qa       = Net1(nn_width,nn_num_layers,True).to(device) 
Best_model_Wia       = Net1(nn_width,nn_num_layers).to(device) 
Best_model_Wha       = Net1(nn_width,nn_num_layers).to(device) 
Best_model_Sigea       = Net1(nn_width,nn_num_layers).to(device)


para_xi         = list(Xi.parameters())  + list(Xh.parameters())+list(Wha.parameters()) +\
                 list(Qa.parameters()) + list(Wia.parameters()) +list(Sigea.parameters())  + list(Mue.parameters())

optimizer_xi    = optim.Adam(para_xi, lr=0.001)
wlgm = 1
epochs = 50
epochs_sub1 = 10002
loss_data_count = 0
model_dir = "models/model1_gammai_1_gammah_2_new_base"
plot_dir = f"{model_dir}/plots/"
output_dir = f"{model_dir}/output/"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
fo = open(f'{model_dir}/loss_fun.csv', "w")
string = "loss_data_count, loss_val, q_val"
fo.write( (string+'\n') )
fo.flush()

set_seeds(0)
min_loss = float('Inf')
for epoch_sub in range(1,epochs_sub1):
        loss_data_count += 1
        eta=TS.sample(params['batchSize'])
        eta_tensor = torch.tensor(eta,dtype=torch.float32,device = device)
        leq1,leq2,leq3,leq4,leq5,leq6,leq7,leq8,leq9,leq10,leq11,lhjb_i,lhjb_h = TP.loss_fun(Xi,Xh,Mue,Sigea,Qa,Wia,Wha,eta_tensor)
        leq1_ =torch.mean(torch.square(leq1)) 
        leq2_ =torch.mean(torch.square(leq2))
        leq3_ =torch.mean(torch.square(leq3))
        leq4_ =torch.mean(torch.square(leq4))
        leq5_ =torch.mean(torch.square(leq5))
        leq6_ =torch.mean(torch.square(leq6))
        leq7_ =torch.mean(torch.square(leq7))
        leq8_ =torch.mean(torch.square(leq8))
        leq9_ =torch.mean(torch.square(leq9))
        leq10_ =torch.mean(torch.square(leq10))
        leq11_ =torch.mean(torch.square(leq11))
        lhjb_i_ = torch.mean(torch.square(lhjb_i))
        lhjb_h_ = torch.mean(torch.square(lhjb_h))
        loss = leq1_+leq2_+leq3_+leq4_+leq5_+leq6_+leq7_+leq8_+leq9_+leq10_+leq11_+lhjb_i_+lhjb_h_
        
        optimizer_xi.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(para_xi, 1)
        optimizer_xi.step()
        loss_val = loss.item()
        if (loss_val < min_loss):
        #if True:
            min_loss = loss_val
            Best_model_Xi.load_state_dict(Xi.state_dict())
            Best_model_Xh.load_state_dict(Xh.state_dict())
            Best_model_Mue.load_state_dict(Mue.state_dict())
            Best_model_Qa.load_state_dict(Qa.state_dict())
            Best_model_Wia.load_state_dict(Wia.state_dict())
            Best_model_Wha.load_state_dict(Wha.state_dict())
            Best_model_Sigea.load_state_dict(Sigea.state_dict())
            print("\n best loss HJB:",min_loss,end="\r")
            print("HJB Block Loss at epoch number:", epoch_sub,"--")
            print(leq1_.item(),leq2_.item(),leq3_.item(),leq4_.item(),leq5_.item(),\
                  leq6_.item(),leq7_.item(),leq8_.item(),leq9_.item(),leq10_.item(),lhjb_i_.item(),lhjb_h_.item())
        
        if (loss_data_count%2000 == 0):
            if True:
                
                eta_grid = np.linspace(params['start_eta'],params['end_eta'],params['neta'])
                e = torch.tensor(eta_grid.reshape(-1,1),dtype=torch.float32,device = device)
                
                #for debugging
                #z = torch.ones((100,1)) * 0.5
                #e = torch.linspace(0.01,0.9,100).reshape(-1,1)
                Xi.load_state_dict(Best_model_Xi.state_dict())
                Xh.load_state_dict(Best_model_Xh.state_dict())
                Mue.load_state_dict(Best_model_Mue.state_dict())
                Qa.load_state_dict(Best_model_Qa.state_dict())
                Wia.load_state_dict(Best_model_Wia.state_dict())
                Wha.load_state_dict(Best_model_Wha.state_dict())
                Sigea.load_state_dict(Best_model_Sigea.state_dict())
                
                
                e.requires_grad_(True)
                
                # reevaluate all endogenous variables and associated variables
                xi   = Xi(e)
                xh   = Xh(e) 
               
                mue = Mue(e)
                sigea = Sigea(e)
                qa = Qa(e)
                wia = Wia(e)
                wha = Wha(e)
                iota_a = (qa-1)/params['kappa']
                Phi_a = 1/params['kappa']*torch.log(1+params['kappa']*iota_a)
                
                ci = (params['rhoi']**params['zetai'])*xi**(1 - params['zetai'])
                ch = (params['rhoh']**params['zetah'])*xh**(1 - params['zetah'])
    
                # automatic differentiations for qa
                qa_e   =  TP.get_derivs_1order(qa,e)[:,0].unsqueeze(1) 
                qa_ee    = TP.get_derivs_1order(qa_e,e)[:,0].unsqueeze(1) 

                # automatic differentiations for xie and xih
                xi_e  = TP.get_derivs_1order(xi,e)[:,0].unsqueeze(1) 
                xi_ee  = TP.get_derivs_1order(xi_e,e)[:,0].unsqueeze(1) 
                
                xh_e  = TP.get_derivs_1order(xh,e)[:,0].unsqueeze(1) 
                xh_ee  = TP.get_derivs_1order(xh_e,e)[:,0].unsqueeze(1) 
                
                sigqaa = qa_e/qa*sigea*e 
                signia = wia*(params['siga']+sigqaa)
                signha = wha*(params['siga']+sigqaa)
                sigxia = xi_e/xi*sigea*e 
                sigxha = xh_e/xh*sigea*e 

                
                muqa = qa_e/qa*mue*e + 1/2*qa_ee/qa*( sigea**2 )*e**2 
                rka  = (params['aa'] - iota_a)/qa + params['mua'] + Phi_a + muqa + sigqaa*params['siga'] 
                r = rka  - params['gammah']*wha*((params['siga'] + sigqaa) )**2\
                               + (1-params['gammah'])*sigxha*((params['siga'] + sigqaa))
                muni = r - ci + wia*(rka- r) 
                munh = r - ch + wha*(rka-r) 
                muqk = (muqa + params['mua'] + Phi_a + params['siga']*sigqaa)  
                
                rp_a = rka-r 
                sig_eta = sigea
                sr_a = (rp_a )/((params['siga'] +sigqaa)**2)**0.5
                sig_a = sigqaa
                qa_np = qa.detach().cpu().numpy().reshape(-1)
                data = np.hstack([eta_grid.reshape(-1,1),ci.detach().cpu().numpy(),sr_a.detach().cpu().numpy(),rp_a.detach().cpu().numpy(),\
                                      qa.detach().cpu().numpy(),wia.detach().cpu().numpy(),\
                                      sigqaa.detach().cpu().numpy(),\
                                          r.detach().cpu().numpy(),mue.detach().cpu().numpy(),sigea.detach().cpu().numpy(),\
                                              muqa.detach().cpu().numpy(),muni.detach().cpu().numpy(),munh.detach().cpu().numpy(),muqk.detach().cpu().numpy(),sigxia.detach().cpu().numpy(),\
                                                  sigxha.detach().cpu().numpy(),rka.detach().cpu().numpy(),wha.detach().cpu().numpy()])
                data = pd.DataFrame(data)
                data.columns=['eta','ci','sr_a','rp_a','qa','wia','sigqaa','r','mue','sigea',\
                              'muqa','muni','munh','muqk','sigxia','sigxha','rka','wha']
                data.to_csv(f'{output_dir}/output_' + str(params['gammah']) +'_' + str(params['muO'])  + '.csv')
                with open(f'{output_dir}/params_' + str(params['gammah']) +'_' + str(params['muO'])  + '.txt', 'w') as file:
                     file.write(json.dumps(params))
                

            eta_plt = eta_grid
            fig, ax = plt.subplots(4, 2, figsize=(12, 24))
            ax[0,0].plot(eta_plt,xi.detach().cpu().numpy())
            ax[0,0].set_ylabel('xii')
            ax[0,0].set_xlabel("eta")
            ax[0,1].plot(eta_plt,xh.detach().cpu().numpy())
            ax[0,1].set_ylabel('xih')
            ax[0,1].set_xlabel("eta")

            ax[1,0].plot(eta_plt,mue.detach().cpu().numpy())
            ax[1,0].set_ylabel(r'$\mu^{\eta}$')
            ax[1,0].set_xlabel("eta")
            ax[1,1].plot(eta_plt,sigea.detach().cpu().numpy())
            ax[1,1].set_ylabel(r'$\sigma^{\eta}$')
            ax[1,1].set_xlabel("eta")
        
            ax[2,0].plot(eta_plt,qa.detach().cpu().numpy())
            ax[2,0].set_ylabel('qa')
            ax[2,0].set_xlabel("eta")

            ax[3,0].plot(eta_plt,wia.detach().cpu().numpy())
            ax[3,0].set_ylabel('wia')
            ax[3,0].set_xlabel("eta")
            ax[3,1].plot(eta_plt,wha.detach().cpu().numpy())
            ax[3,1].set_ylabel('wha')
            ax[3,1].set_xlabel("eta")

            name = plot_dir + 'plot_train_' + str(loss_data_count)+'_1.png'
            plt.tight_layout()
            plt.savefig(name,bbox_inches='tight',dpi=300)
            plt.close('all')

            fig, ax = plt.subplots(2,4,figsize=(16,9), num=0)
            ax[0,0].plot(eta_plt,rka.detach().cpu().numpy())
            ax[0,0].set_ylabel('rka')
            ax[0,0].legend(fontsize=15)
            ax[0,1].plot(eta_plt,rp_a.detach().cpu().numpy())
            ax[0,1].set_ylabel('rp_a')
            ax[0,2].plot(eta_plt,qa.detach().cpu().numpy())
            ax[0,2].set_ylabel('qa')
            ax[0,3].plot(eta_plt,r.detach().cpu().numpy())
            ax[0,3].set_ylabel('r')
            ax[1,0].plot(eta_plt,wha.detach().cpu().numpy())
            ax[1,0].set_ylabel('wha')
            ax[1,0].set_xlabel('$\eta$')
            ax[1,1].plot(eta_plt,sigqaa.detach().cpu().numpy())
            ax[1,1].set_ylabel('sigqaa')
            ax[1,1].set_xlabel(r'$\eta$')
            ax[1,2].plot(eta_plt,eta_plt.reshape(-1,1)*sigea.detach().cpu().numpy())
            ax[1,2].set_ylabel(r'$\eta \sigma^{\eta}$')
            ax[1,2].set_xlabel('$\eta$')
            ax[1,3].plot(eta_plt,eta_plt.reshape(-1,1)*mue.detach().cpu().numpy())
            ax[1,3].set_ylabel(r'$\eta \mu^{\eta}$')
            ax[1,3].set_xlabel(r'$\eta$')

            name = plot_dir + 'plot_train_' + str(loss_data_count)+'_2.png'
            plt.tight_layout()
            plt.savefig(name,bbox_inches='tight',dpi=300)
            plt.close('all')
            

            fo = open(f'{model_dir}/loss_fun.csv', "a")
            string = "%.4e,%.4e" % (np.log(loss_data_count), np.log(loss_val))
            fo.write( (string+'\n') )
            fo.flush()


print("{0:=^80}".format("Running L-BFGS optimization"))
Xi.load_state_dict(Best_model_Xi.state_dict())
Xh.load_state_dict(Best_model_Xh.state_dict())
Mue.load_state_dict(Best_model_Mue.state_dict())
Qa.load_state_dict(Best_model_Qa.state_dict())
Wia.load_state_dict(Best_model_Wia.state_dict())
Wha.load_state_dict(Best_model_Wha.state_dict())
Sigea.load_state_dict(Best_model_Sigea.state_dict())

optimizer_lbfgs = torch.optim.LBFGS(para_xi, lr=1)
epochs_lbfgs = 102

def closure(eta):
    optimizer_lbfgs.zero_grad()
    eta=TS.sample(params['batchSize'])
    eta_tensor = torch.tensor(eta,dtype=torch.float32,device = device)
    leq1,leq2,leq3,leq4,leq5,leq6,leq7,leq8,leq9,leq10,leq11,lhjb_i,lhjb_h = TP.loss_fun(Xi,Xh,Mue,Sigea,Qa,Wia,Wha,eta_tensor)
    leq1_ =torch.mean(torch.square(leq1)) 
    leq2_ =torch.mean(torch.square(leq2))
    leq3_ =torch.mean(torch.square(leq3))
    leq4_ =torch.mean(torch.square(leq4))
    leq5_ =torch.mean(torch.square(leq5))
    leq6_ =torch.mean(torch.square(leq6))
    leq7_ =torch.mean(torch.square(leq7))
    leq8_ =torch.mean(torch.square(leq8))
    leq9_ =torch.mean(torch.square(leq9))
    leq10_ =torch.mean(torch.square(leq10))
    leq11_ =torch.mean(torch.square(leq11))
    lhjb_i_ = torch.mean(torch.square(lhjb_i))
    lhjb_h_ = torch.mean(torch.square(lhjb_h))
    loss = leq1_+leq2_+leq3_+leq4_+leq5_+leq6_+leq7_+leq8_+leq9_+leq10_+leq11_+lhjb_i_+lhjb_h_
    loss.backward()
    return loss

set_seeds(0)
for epoch_sub in range(1, epochs_lbfgs):
    loss_data_count += 1
    
    optimizer_lbfgs.step(lambda: closure(eta))

    leq1,leq2,leq3,leq4,leq5,leq6,leq7,leq8,leq9,leq10,leq11,lhjb_i,lhjb_h = TP.loss_fun(Xi,Xh,Mue,Sigea,Qa,Wia,Wha,eta_tensor)
    leq1_ =torch.mean(torch.square(leq1)) 
    leq2_ =torch.mean(torch.square(leq2))
    leq3_ =torch.mean(torch.square(leq3))
    leq4_ =torch.mean(torch.square(leq4))
    leq5_ =torch.mean(torch.square(leq5))
    leq6_ =torch.mean(torch.square(leq6))
    leq7_ =torch.mean(torch.square(leq7))
    leq8_ =torch.mean(torch.square(leq8))
    leq9_ =torch.mean(torch.square(leq9))
    leq10_ =torch.mean(torch.square(leq10))
    leq11_ =torch.mean(torch.square(leq11))
    lhjb_i_ = torch.mean(torch.square(lhjb_i))
    lhjb_h_ = torch.mean(torch.square(lhjb_h))
    loss = leq1_+leq2_+leq3_+leq4_+leq5_+leq6_+leq7_+leq8_+leq9_+leq10_+leq11_+lhjb_i_+lhjb_h_
    
    loss_val = loss.item()
    if (loss_val < min_loss):
    #if True:
        min_loss = loss_val
        Best_model_Xi.load_state_dict(Xi.state_dict())
        Best_model_Xh.load_state_dict(Xh.state_dict())
        Best_model_Mue.load_state_dict(Mue.state_dict())
        Best_model_Qa.load_state_dict(Qa.state_dict())
        Best_model_Wia.load_state_dict(Wia.state_dict())
        Best_model_Wha.load_state_dict(Wha.state_dict())
        Best_model_Sigea.load_state_dict(Sigea.state_dict())
        print("\n best loss HJB:",min_loss,end="\r")
        print("HJB Block Loss at epoch number:", epoch_sub,"--")
        print(leq1_.item(),leq2_.item(),leq3_.item(),leq4_.item(),leq5_.item(),\
                leq6_.item(),leq7_.item(),leq8_.item(),leq9_.item(),leq10_.item(),lhjb_i_.item(),lhjb_h_.item())
        
    if (loss_data_count%100 == 0):
        if True:
            
            eta_grid = np.linspace(params['start_eta'],params['end_eta'],params['neta'])
            e = torch.tensor(eta_grid.reshape(-1,1),dtype=torch.float32,device = device)
            
            #for debugging
            #z = torch.ones((100,1)) * 0.5
            #e = torch.linspace(0.01,0.9,100).reshape(-1,1)
            Xi.load_state_dict(Best_model_Xi.state_dict())
            Xh.load_state_dict(Best_model_Xh.state_dict())
            Mue.load_state_dict(Best_model_Mue.state_dict())
            Qa.load_state_dict(Best_model_Qa.state_dict())
            Wia.load_state_dict(Best_model_Wia.state_dict())
            Wha.load_state_dict(Best_model_Wha.state_dict())
            Sigea.load_state_dict(Best_model_Sigea.state_dict())
            
            
            e.requires_grad_(True)
            
            # reevaluate all endogenous variables and associated variables
            xi   = Xi(e)
            xh   = Xh(e) 
            
            mue = Mue(e)
            sigea = Sigea(e)
            qa = Qa(e)
            wia = Wia(e)
            wha = Wha(e)
            iota_a = (qa-1)/params['kappa']
            Phi_a = 1/params['kappa']*torch.log(1+params['kappa']*iota_a)
            
            ci = (params['rhoi']**params['zetai'])*xi**(1 - params['zetai'])
            ch = (params['rhoh']**params['zetah'])*xh**(1 - params['zetah'])

            # automatic differentiations for qa
            qa_e   =  TP.get_derivs_1order(qa,e)[:,0].unsqueeze(1) 
            qa_ee    = TP.get_derivs_1order(qa_e,e)[:,0].unsqueeze(1) 

            # automatic differentiations for xie and xih
            xi_e  = TP.get_derivs_1order(xi,e)[:,0].unsqueeze(1) 
            xi_ee  = TP.get_derivs_1order(xi_e,e)[:,0].unsqueeze(1) 
            
            xh_e  = TP.get_derivs_1order(xh,e)[:,0].unsqueeze(1) 
            xh_ee  = TP.get_derivs_1order(xh_e,e)[:,0].unsqueeze(1) 
            
            sigqaa = qa_e/qa*sigea*e 
            signia = wia*(params['siga']+sigqaa)
            signha = wha*(params['siga']+sigqaa)
            sigxia = xi_e/xi*sigea*e 
            sigxha = xh_e/xh*sigea*e 

            
            muqa = qa_e/qa*mue*e + 1/2*qa_ee/qa*( sigea**2 )*e**2 
            rka  = (params['aa'] - iota_a)/qa + params['mua'] + Phi_a + muqa + sigqaa*params['siga'] 
            r = rka  - params['gammah']*wha*((params['siga'] + sigqaa) )**2\
                            + (1-params['gammah'])*sigxha*((params['siga'] + sigqaa))
            muni = r - ci + wia*(rka- r) 
            munh = r - ch + wha*(rka-r) 
            muqk = (muqa + params['mua'] + Phi_a + params['siga']*sigqaa)  
            
            rp_a = rka-r 
            sig_eta = sigea
            sr_a = (rp_a )/((params['siga'] +sigqaa)**2)**0.5
            sig_a = sigqaa
            qa_np = qa.detach().cpu().numpy().reshape(-1)
            data = np.hstack([eta_grid.reshape(-1,1),ci.detach().cpu().numpy(),sr_a.detach().cpu().numpy(),rp_a.detach().cpu().numpy(),\
                                    qa.detach().cpu().numpy(),wia.detach().cpu().numpy(),\
                                    sigqaa.detach().cpu().numpy(),\
                                        r.detach().cpu().numpy(),mue.detach().cpu().numpy(),sigea.detach().cpu().numpy(),\
                                            muqa.detach().cpu().numpy(),muni.detach().cpu().numpy(),munh.detach().cpu().numpy(),muqk.detach().cpu().numpy(),sigxia.detach().cpu().numpy(),\
                                                sigxha.detach().cpu().numpy(),rka.detach().cpu().numpy(),wha.detach().cpu().numpy()])
            data = pd.DataFrame(data)
            data.columns=['eta','ci','sr_a','rp_a','qa','wia','sigqaa','r','mue','sigea',\
                            'muqa','muni','munh','muqk','sigxia','sigxha','rka','wha']
            data.to_csv(f'{output_dir}/output_' + str(params['gammah']) +'_' + str(params['muO'])  + '.csv')
            with open(f'{output_dir}/params_' + str(params['gammah']) +'_' + str(params['muO'])  + '.txt', 'w') as file:
                    file.write(json.dumps(params))
            

        eta_plt = eta_grid
        fig, ax = plt.subplots(4, 2, figsize=(12, 24))
        ax[0,0].plot(eta_plt,xi.detach().cpu().numpy())
        ax[0,0].set_ylabel('xii')
        ax[0,0].set_xlabel("eta")
        ax[0,1].plot(eta_plt,xh.detach().cpu().numpy())
        ax[0,1].set_ylabel('xih')
        ax[0,1].set_xlabel("eta")

        ax[1,0].plot(eta_plt,mue.detach().cpu().numpy())
        ax[1,0].set_ylabel(r'$\mu^{\eta}$')
        ax[1,0].set_xlabel("eta")
        ax[1,1].plot(eta_plt,sigea.detach().cpu().numpy())
        ax[1,1].set_ylabel(r'$\sigma^{\eta}$')
        ax[1,1].set_xlabel("eta")
    
        ax[2,0].plot(eta_plt,qa.detach().cpu().numpy())
        ax[2,0].set_ylabel('qa')
        ax[2,0].set_xlabel("eta")

        ax[3,0].plot(eta_plt,wia.detach().cpu().numpy())
        ax[3,0].set_ylabel('wia')
        ax[3,0].set_xlabel("eta")
        ax[3,1].plot(eta_plt,wha.detach().cpu().numpy())
        ax[3,1].set_ylabel('wha')
        ax[3,1].set_xlabel("eta")

        name = plot_dir + 'plot_train_' + str(loss_data_count)+'_1.png'
        plt.tight_layout()
        plt.savefig(name,bbox_inches='tight',dpi=300)
        plt.close('all')

        fig, ax = plt.subplots(2,4,figsize=(16,9), num=0)
        ax[0,0].plot(eta_plt,rka.detach().cpu().numpy())
        ax[0,0].set_ylabel('rka')
        ax[0,0].legend(fontsize=15)
        ax[0,1].plot(eta_plt,rp_a.detach().cpu().numpy())
        ax[0,1].set_ylabel('rp_a')
        ax[0,2].plot(eta_plt,qa.detach().cpu().numpy())
        ax[0,2].set_ylabel('qa')
        ax[0,3].plot(eta_plt,r.detach().cpu().numpy())
        ax[0,3].set_ylabel('r')
        ax[1,0].plot(eta_plt,wha.detach().cpu().numpy())
        ax[1,0].set_ylabel('wha')
        ax[1,0].set_xlabel('$\eta$')
        ax[1,1].plot(eta_plt,sigqaa.detach().cpu().numpy())
        ax[1,1].set_ylabel('sigqaa')
        ax[1,1].set_xlabel(r'$\eta$')
        ax[1,2].plot(eta_plt,eta_plt.reshape(-1,1)*sigea.detach().cpu().numpy())
        ax[1,2].set_ylabel(r'$\eta \sigma^{\eta}$')
        ax[1,2].set_xlabel('$\eta$')
        ax[1,3].plot(eta_plt,eta_plt.reshape(-1,1)*mue.detach().cpu().numpy())
        ax[1,3].set_ylabel(r'$\eta \mu^{\eta}$')
        ax[1,3].set_xlabel(r'$\eta$')

        name = plot_dir + 'plot_train_' + str(loss_data_count)+'_2.png'
        plt.tight_layout()
        plt.savefig(name,bbox_inches='tight',dpi=300)
        plt.close('all')
        

        fo = open(f'{model_dir}/loss_fun.csv', "a")
        string = "%.4e,%.4e" % (np.log(loss_data_count), np.log(loss_val))
        fo.write( (string+'\n') )
        fo.flush()