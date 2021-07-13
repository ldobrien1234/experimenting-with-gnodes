# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:48:18 2021

Liam O'Brien

This code runs a graph neural ordinary differential equation to learn
the trajectories of five stars/planets given their initial mass,
position, and velocity.
"""
import pickle
import time
import math
import matplotlib.pyplot as plt

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.linalg as la
import torch.nn as nn
import torch.nn.functional as F

from GCNLayers import GCNLayer1, GNODEFunc, ODEBlock
from planets_dataset import PlanetDataset #so pickle can find the class


#lets us use our computer's gpu if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#seed for repeatability
#ensures only deterministic convolution algorithms
torch.backends.cudnn.deterministic = True
#does not look for fasest convolution algorithm
torch.backends.cudnn.benchmark = False

torch.manual_seed(0) #seed for random numbers




#gets our data from the file "data.p"
file = open("mini.p", "rb")
dataset = pickle.load(file)
file.close()


def get_features(graphs):
    """
    Given a list of dgl graphs, this function returns a tensor of the
    features of the input graphs.
    
    Parameters
    ----------
    graphs: list
        graphs is a list of dgl graphs.
        
    Returns
    -------
    features: list
        features is a list of 2 dimensional tensors that are the 
        feature matrix for each graph in the input.
    """
    features = []
    for graph in graphs:
        feature = graph.ndata['h']
        features += [feature]
    
    return features
        
    
    
#creating our training inputs and targets (tensor of feature matrices)
X_train = get_features(dataset.train_inputs())
X_train = torch.stack(X_train).float() #convert list to tensor
Y_train = get_features(dataset.train_targets())
Y_train = torch.stack(Y_train).float() #convert list to tensor

#creating our testing inputs and targets (tensor of feature matrices)
X_test = get_features(dataset.test_inputs())
X_test = torch.stack(X_test).float() #convert list to tensor
Y_test = get_features(dataset.test_targets())
Y_test = torch.stack(Y_test).float #convert list to tensor



#get the structure of the graph
g_top = dataset.train_inputs()[0]

#dynamics defined by two GCN layers
gnn = nn.Sequential(GCNLayer1(g=g_top, in_feats=7, out_feats=40, 
                              dropout=0.5, activation=nn.Softplus()),
                  GCNLayer1(g=g_top, in_feats=40, out_feats=7, 
                            dropout=0.5, activation=None)
                  ).to(device)

gnode_func = GNODEFunc(gnn)


#ODEBlock class let's us use an ode solver to find our input at a later time
#ode too stiff for adaptive solvers
gnode = ODEBlock(gnode_func, method = 'implicit_adams', atol=1e-3, rtol=1e-4,
                 adjoint = True) 
model = gnode

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.L1Loss()


train_steps = len(X_train)
test_steps = len(X_test)
minibatch = 10 #size of minibatch
train_n_batches = int(train_steps / minibatch)
test_n_batches = int(test_steps / minibatch)

num_grad_steps = 0

train_MAPE_plt = []
train_y_plt = []
#running the training loop

for batch in range(train_n_batches):
    model.train() #modifies forward for training (e.g. performs dropout)
        
    model.gnode_func.nfe = 0 #zero nfe for next forward pass

    batch_X = X_train[batch*minibatch:(batch+1)*minibatch]
    batch_Y = Y_train[batch*minibatch:(batch+1)*minibatch]
    
    MAPEs = []
    losses = []
    for observation in range(minibatch):
            
        start_time = time.time()
        y_pred = model(batch_X[observation])
        f_time = time.time() - start_time #store time for forward pass
        
        nfe = model.gnode_func.nfe #store nfe in forward pass
            
        y = batch_Y[observation]
        
        MAPE = la.norm((torch.abs(y - y_pred) / torch.abs(y)), 
                       ord=2).detach().item()
        loss = criterion(y_pred,y)
            
        losses.append(loss)
        MAPEs.append(MAPE)
        
    MAPE_avg = sum(MAPEs)  / len(MAPEs)
    loss_avg = sum(losses) / len(losses) #averaging over the minibatch
    
        
    opt.zero_grad()
        
    start_time = time.time()
    loss_avg.backward()
    b_time = time.time() - start_time #store time for backward pass
    
    #avoid exploding gradient
    nn.utils.clip_grad_norm_(model.parameters(), 1e+5)
        
    opt.step()
    num_grad_steps += 1
    
    train_loss = loss_avg.item()
    
    print("MAPE = ", MAPE_avg)
    print("train loss = ", train_loss)
    if math.isnan(train_loss):
        break
    train_MAPE_plt.append(MAPE_avg)
    train_y_plt.append(train_loss)
            





#using testing data
with torch.no_grad():
    model.eval() #modifies forward for evaluation (e.g. no dropout)
    
    test_MAPE_plt = []
    test_y_plt = []
    for batch in range(test_n_batches):
        batch_X = X_train[batch*minibatch:(batch+1)*minibatch]
        batch_Y = Y_train[batch*minibatch:(batch+1)*minibatch]
        
        MAPEs = []
        losses = []
        for observation in range(minibatch):
            
            start_time = time.time()
            y_pred = model(batch_X[observation])
            f_time = time.time() - start_time #store time for forward pass
        
            nfe = model.gnode_func.nfe #store nfe in forward pass
            
            y = batch_Y[observation]
            with torch.no_grad():
                MAPE = la.norm((torch.abs(y - y_pred) / torch.abs(y)),
                               ord=2).item()
            loss = criterion(y_pred,y)
            
            MAPEs.append(MAPE)
            losses.append(loss)
        
        MAPE_avg = sum(MAPEs)  / len(MAPEs)
        loss_avg = sum(losses) / len(losses) #averaging over the minibatch
        
        test_loss = loss_avg.item()
        
        print("MAPE = ", MAPE_avg)
        print("test loss = ", test_loss)
        
        test_MAPE_plt.append(MAPE_avg)
        test_y_plt.append(test_loss)
    
    
     
train_x_plt = np.array(range(len(train_y_plt)))       
train_y_plt = np.array(train_y_plt)

test_x_plt = np.array(range(len(test_y_plt)))
test_y_plt = np.array(test_y_plt)

plt.plot(train_x_plt, train_y_plt, 'blue', label='training loss')
plt.plot(test_x_plt, test_y_plt, 'orange',label='testing loss')
plt.title('Loss')
plt.legend()

       
#making a new plot for MAPE 
plt.figure()

train_MAPE_plt = np.array(train_MAPE_plt)
test_MAPE_plt = np.array(test_MAPE_plt)

plt.plot(train_x_plt, train_MAPE_plt, 'blue', label='training MAPE')
plt.plot(test_x_plt, test_MAPE_plt, 'orange', label='testing MAPE')
plt.title('MAPE')
plt.legend()
