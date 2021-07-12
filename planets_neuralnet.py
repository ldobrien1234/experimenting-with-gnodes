# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:48:18 2021

Liam O'Brien
"""
import pickle
import time

import dgl
import dgl.function as fn

import torch
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
file = open("fix_t_data.p", "rb")
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
        
    
    
#creating our training inputs and  (tensor of feature matrices)
X_train = get_features(dataset.train_inputs())
Y_train = get_features(dataset.train_targets())

#get the structure of the graph
g_top = dataset.train_inputs()[0]

#dynamics defined by two GCN layers
gnn = nn.Sequential(GCNLayer1(g=g_top, in_feats=7, out_feats=16, 
                              activation=nn.Softplus(), dropout=0.9),
                  GCNLayer1(g=g_top, in_feats=16, out_feats=7, 
                            activation=None, dropout=0.9)
                 ).to(device)

gnode_func = GNODEFunc(gnn)

#ODEBlock class let's us use an ode solver to find our input at a later time

gnode = ODEBlock(gnode_func, method = 'rk4') #ode too stiff for adaptive solvers
model = gnode

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.L1Loss()


steps = len(X_train) #updating weights after each training example
verbose_step = 300 #number of steps after which we print information
num_grad_steps = 0



#running the training loop
for step in range(steps):
    model.train() #modifies forward for training (performs dropout)
    
    model.gnode_func.nfe = 0 #zero nfe for next forward pass
    
    start_time = time.time()
    y_pred = model(X_train[step].float())
    f_time = time.time() - start_time #store time for forward pass
    nfe = model.gnode_func.nfe #store nfe in forward pass
    
    y = Y_train[step]
    
    # print(torch.isnan(X_train[step].float()))
    # print(torch.isnan(y_pred))
    # print(torch.isnan(y))
    
    
    loss = criterion(y_pred,y)
    opt.zero_grad()
    
    start_time = time.time()
    loss.backward()
    b_time = time.time() - start_time #store time for backward pass
    
    opt.step()
    num_grad_steps += 1
    
    print("train loss = ", loss.item())

        
    
    


        
        




