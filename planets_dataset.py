# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:40:27 2021

@author: Liam O'Brien

This script creates a dataset of graphs with input and output pairs
for graph machine learning. The inputs represent the state of five
planets at time 0, and the target is the state of the planets at some
final time. The latent vector associated with the initial node gives the
mass, position, and velocity. Each graph is complete and the edges are
weighted based on Euclidean distance. The dataset is exported to a file
using Python's pickle module.


"""

import planets_simulation as plan

import torch
import numpy as np
import numpy.random as nprand
import math
import time

import dgl
from dgl.data import DGLDataset

import pickle



#Creating DGLDataset class

class PlanetDataset(DGLDataset):
    """
    This class allows us to use methods to get our training set and 
    test set from our data
    
    Parameters
    ----------
    init_graphs: a list of DGLGraphs
        init_graphs should be a list of your input DGLGraphs.
    fin_graphs: a list of DGLGraphs
        fin_graphs should be a list of your target DGLGraphs.
    
    """
    
    def __init__(self, init_graphs,fin_graphs,):
        
        self.inputs = init_graphs #list of input graphs
        self.targets = fin_graphs #list of target output graph
        
    def size(self):
        #returns the number of input, target pairs in our dataset
        return len(self.inputs)
    
    def train_inputs(self):
        #gives us a portion of the input data for training
        #want 60% of the data to be training set
        num_examples = math.ceil(self.size()*6 / 10)
            
        #slicing the list of inputs
        return self.inputs[:num_examples]
        
    def train_targets(self):
        #gives us a portion of the target data for training
        #want 60% of the data to be training set
        num_examples = math.ceil(self.size()*6 / 10)
            
        #slicing the list of inputs
        return self.targets[:num_examples]
        
    def test_inputs(self):
        #gives us the remaining input data for testing
        num_examples = math.ceil(self.size()*6 / 10)
            
        #slicing list of inputs
        return self.inputs[num_examples:]
        
    def test_targets(self):
        #gives us the remaining target data for testing
        num_examples = math.ceil(self.size()*6 / 10)
            
        #slicing list of targets
        return self.inputs[:num_examples]
            
    

SIZE_DATASET = 20000 #of input, target pairs we want to create
MAX_TIME = 60*60*24*365*10**5 # max final time in seconds


#creating random final times for each input, target pair
nprand.seed() #seed for repeatability
final_times = nprand.uniform(0, MAX_TIME, SIZE_DATASET)


def feat_matrix(f_time):
    """
    Given the final time of our planets, we can create two feature matrices
    that have the final time, mass, position, and velocity for each 
    planet.
    
    Parameters
    ----------
    f_time: float
        f_time is the final time at which we want to know the position of
        the planets.
    Returns
    -------
    A tuple of the initial feature matrix and the final feature matrix.
    """
    (final, MASS) = plan.simulation(T = f_time)
    
    MASS = torch.from_numpy(MASS)
    MASS = MASS.reshape(5,1) #makes into column vector
    
    #we want to make the final time part of the feature matrix
    #so the neuralnet can learn variable time data
    TIME = torch.tensor([f_time]*5)
    TIME = TIME.reshape(5,1)
    
    #Finding the initial feature matrix of positions and velocities
    np_init_h = final.y[::,0] #initial position, velocity
    np_init_h = np.reshape(np_init_h, (5,6)) #changes from vector to matrix
    #converts to tensor, appends mass in 0th column, new assignment
    initial_h = torch.cat((TIME, MASS, torch.from_numpy(np_init_h)), 1) 
    
    np_fin_h = final.y[::,-1] #final position, velocity
    np_fin_h = np.reshape(np_fin_h, (5,6)) #changes from vector to matrix
    #converts to tensor, appends mass in 0th column, new assignmnet
    final_h = torch.cat((TIME, MASS, torch.from_numpy(np_fin_h)), 1) 
    
    return (initial_h, final_h)

def edge_features(init_positions, fin_positions):
    """
    Given a matrix of initial and final positions of our planets, this
    function can give us the edge features and relationships, so we
    can construct a graph using dgl.heterograph.
    
    Parameters
    ----------
    init_positions: (5,3) torch tensor
        init_positions is a 5 x 3 matrix representing the initial
        position of each planet. Each row is a planet and the row
        vectors represent the position.
    fin_positions: (5,3) torch tensor
        fin_positions is a 5 x 3 matrix representing the initial
        position of each planet. Each row is a planet and the row
        vectors represent the position.
        
    Returns
    -------
    edge_features returns a 4-tuple of the initial edge feature matrix,
    the final edge feature matrix, the first tensor to pass into
    dgl.heterograph, and the second tensor to pass into dgl.heterograph.

    """
    
    #initialize edge feature matrix
    init_distances = torch.empty(0)
    fin_distances = torch.empty(0)
    
    #creating tensors defining adjacency to pass to dgl.heterograph
    tensor1 = torch.empty(0, dtype = torch.int)
    tensor2 = torch.empty(0, dtype = torch.int)
    
    #iterating through the source node/planet
    for source in range(5):
        
        #iterating through the destination node/planet
        for destination in range(5):
            
            init_distance = math.dist(init_positions[source,::], 
                                      init_positions[destination,::])
            init_distances = torch.cat((init_distances,
                                       torch.tensor([init_distance])))
            
            fin_distance = math.dist(fin_positions[source,::], 
                                      fin_positions[destination,::])
            fin_distances = torch.cat((fin_distances, 
                                      torch.tensor([fin_distance])))
            
            tensor1 = torch.cat((tensor1,torch.tensor([source],
                                                      dtype = torch.int)))
            tensor2 = torch.cat((tensor2, torch.tensor([destination],
                                                       dtype = torch.int)))
    init_distances = torch.reshape(init_distances, (5**2,1))
    fin_distances = torch.reshape(fin_distances, (5**2,1))
    
    return (init_distances, fin_distances, tensor1, tensor2)


def create_dataset():
    """
    create_dataset is a function that creates two lists of graphs where
    corresponding indeces are input, target pairs. The graphs represent
    the inital and final state of a system of five planets

    Returns
    -------
    create_dataset returns a tuple of lists. The first list contains graphs
    that represent the initial state of a system of five planets, and the
    second list contains the corresponding graphs that represent the final
    state of the system.
    """
    
    #create empty lists that we will grow
    init_graphs = []
    fin_graphs = []
    
    start_time = time.time()
    #create a initial and final feature matrix for each datapoint
    for data_index in range(SIZE_DATASET):
        
        #get a final time
        f_time = final_times[data_index] #getting random final time
        #complete a simulation of planets up to the final time
        (initial_h, final_h) = feat_matrix(f_time)
        
        #now that we have the feature matrices, we want the graph structure
        #I will use a complete graph with weighted edges based on distance
        init_positions = initial_h[::, 2:5]
        fin_positions = final_h[::, 2:5]
        
        (init_distances,
         fin_distances,
         tensor1,
         tensor2) = edge_features(init_positions, fin_positions)
        
        #creating a complete graph with self-loops for structure
        g_form = dgl.heterograph({('planet', 'distance', 'planet'): 
                                  (tensor1, tensor2)})
        
        #creating initial graph
        g_init = g_form
        #assign feature matrix
        g_init.ndata['h'] = initial_h
        g_init.edata['dist'] = init_distances
       
        
        #creating final graph
        g_fin = g_form
        #assign feature matrix
        g_fin.ndata['h'] = final_h
        g_fin.edata['dist'] = fin_distances
        
        #appending g_init and g_fin to our list of graph data
        init_graphs.append(g_init)
        fin_graphs.append(g_fin)
        
        #to see progress while the code is running
        if data_index % 1000 == 0:
            print(data_index, " graphs were created")
            run_time = time.time() - start_time
            print("The program has run for ", run_time, " seconds.")
        
    return(init_graphs, fin_graphs)
    


#If we run this code, we will get a graph dataset, and it will be saved
if __name__ == "__main__":
    
    (init_graphs, fin_graphs) = create_dataset()
    
    dataset = PlanetDataset(init_graphs, fin_graphs)
    
        
    
    #save our data to a file named "data.p"
    file = open("t_feat_data.p", "wb")
    pickle.dump(dataset, file)
    file.close()

    
    
    
    
    
    
  
   
