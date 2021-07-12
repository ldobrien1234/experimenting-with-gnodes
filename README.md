# experimenting-with-gnodes
I am creating a Graph Neural Ordinary Differential Equation (GNODE) for modeling Newtonian gravity.

I decided to create a dataset that models a system of interacting planets/stars. The planets_simulation.py program randomizes the initial mass, 
position, and velocity of five planets. Newtonian gravity gives a second order ODE for the position, and the code uses numerical integration to 
solve for the final mass, position, and velocity of the planets at some specified final time. If you run the code, it will also provide graphs of 
the planets’ trajectories.

The second file, planets_dataset.py, runs the planets_simulation.py a specified number of times to create a graph dataset. In the graphs, the planets 
are represented by nodes. Each graph is complete, the nodes have self-loops, and the edges are weighted with Euclidean distance between the planets. 
The node features are a 5 x 7 matrix where each row represents a planet. The first column gives the mass; columns 2-4 give the x, y, and z 
coordinates; and columns 5-7 give the x, y, and z components of velocity. The input and target graphs are contained in two lists, which are 
serialized and exported to a file. I have been using the file fix_t_data.p (with 100,000 observations) to test my neural network. Here, I
included a smaller dataset mini.p, which has 1,000 observations.

Lastly, planets_neuralnet.py will be a GNODE network that will hopefully learn Newtonian gravity. The code uses GCNLayers.py as a module. 
Currently, the model is outputting nan (“not a number”), so I am playing with the hyperparameters to get it to work.
