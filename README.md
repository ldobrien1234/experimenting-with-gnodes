# experimenting-with-gnodes
I am creating a Graph Neural Ordinary Differential Equation (GNODE) for modeling Newtonian gravity.

I decided to create a dataset that models a system of interacting planets/stars. The planets_simulation.py program randomizes the initial mass, 
position, and velocity of five planets. Newtonian gravity gives a second order ODE for the position, and the code uses numerical integration to 
solve for the final mass, position, and velocity of the planets at some specified final time. If you run the code, it will also provide graphs of 
the planets’ trajectories.

The second file, planets_dataset.py, runs the planets_simulation.py a specified number of times to create a graph dataset. In the graphs, the planets 
are represented by nodes. Each graph is complete, the nodes have self-loops, and the edges are weighted with Euclidean distance between the planets. 
The node features are a 5 x 8 matrix where each row represents a planet. The first column gives the final time; the second gives the mass; columns 2-4 
give the x, y, and z coordinates; and columns 5-7 give the x, y, and z components of velocity. The input and target graphs are contained in two lists, 
which are serialized and exported to a file.

Lastly, planets_neuralnet.py is a GNODE that learns the final mass, position, and velocity of the planets given an initial feature matrix (like the
one described above).


Citations:


@article{chen2021eventfn,
  title={Learning Neural Event Functions for Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Amos, Brandon and Nickel, Maximilian},
  journal={International Conference on Learning Representations},
  year={2021}
}


@article{chen2018neuralode,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  journal={Advances in Neural Information Processing Systems},
  year={2018}
}


@article{poli2019graph,
  title={Graph Neural Ordinary Differential Equations},
  author={Poli, Michael and Massaroli, Stefano and Park, Junyoung and Yamashita, Atsushi and Asama, Hajime and Park, Jinkyoo},
  journal={arXiv preprint arXiv:1911.07532},
  year={2019}
}
