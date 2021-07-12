# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:51:47 2021

@author: Liam O'Brien

This script uses numerical methods to solve for the position of a 
system of 5 planets (with randomized masses, positions, and velocities)
at some final time T.
"""

import random as rand
import math

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt


G = 6.7*10**(-11) #Gravitational constant
NUM_PLANETS = 5 #The number of planets that we want to simulate
SOLAR_MASS = 1.989*10**30 #mass of the sun in kg
LIGHT_YEAR = 9.4607*10**15 #light year in m
t_f = 60*60*24*365*10**5 #final time (seconds)



#Creates a random set of planets and specify their masses, 
#position, and velocity
def initialize():
    """
    This function creates the initial conditions for our planet.

    Returns
    -------
    MASS : numpy array
        MASS is a 1D numpy array of size #planets that gives us the mass
        of each planet. It should have the form [m1,. . . ,mn]
        where n is the # of planets.

    COORD : numpy array
        COORD is a flattened numpy array of size #planets*6 
        that gives us the position and velocity of each planet. 
        It should have the form [x1,y1,z1,vx1,vy1,vz1, . . . ,
        xn,yn,zn,vxn,vyn,vzn]] where n is the # of planets.

    """
    COORD = np.empty(0)
    MASS = np.empty(0)
    for i in range(NUM_PLANETS):
        #I played with the ranges to find simulations that were interesting
        #and non-linear
        mass = rand.uniform(0.1*SOLAR_MASS,8*SOLAR_MASS)
        
        x,y,z = (rand.uniform(-0.2*LIGHT_YEAR,0.2*LIGHT_YEAR),
                rand.uniform(-0.2*LIGHT_YEAR,0.2*LIGHT_YEAR),
                rand.uniform(-0.2*LIGHT_YEAR,0.2*LIGHT_YEAR))
        
        vx,vy,vz = (rand.uniform(-10**3,10**3),
                rand.uniform(-5**4,10**3),
                rand.uniform(-10**3,10**3))
        
        #units are kg, m, and m/s
        #use a long vector because solve_ivp only takes 1-D inital values
        MASS = np.append(MASS, (mass))
        COORD = np.append(COORD, (x,y,z, vx,vy,vz))
    
    return (MASS, COORD)


# #mass and coordinates of sun, mercury, venus, earth, 
# #in kg, m, and m/s
# MASS = np.array([1.989*10**(30),
#                   0.330*10**(24),
#                   4.87*10**(24),
#                   5.97*10**(24),
#                   0.642*10**(24)])
# COORD = np.array([0,0,0,0,0,0,
#                     57.9*10**9, 0, 0, 0, 47.4*10**3, 0,
#                     108.2*10**9, 0, 0, 0, 35.0*10**3, 0,
#                     149.6*10**9, 0, 0, 0, 29.8*10**3, 0,
#                     227.9*10**9, 0, 0, 0, 24.1*10**3, 0])

#We can consider a classical description of gravity
def grav_accel(position1, position2, mass2):
    """
    This computes the gravitational acceleration on planet1 from planet2

    Parameters
    ----------
    position1 : numpy array
        numpy array of size (1,3) giving the position vector of planet1
    position2 : numpy array
        numpy array of size (1,3) giving the position vector of planet2
    mass2: float
        float giving the mass of planet2 in

    Returns
    -------
    accel: numpy array
        accel is a numpy array of size (1,3) of float-values 
        representing the gravitational acceleration on the first 
        planet from the second in m/s^2.
        
    """
    
    distance = math.dist(position1, position2)
    
    #If our planets are in the same position, there is no grav force
    if distance == 0:
        return 0
    else:
        
        #perform elementwise subtraction
        direction = np.subtract(position2,position1)
        unit_direction = direction/(math.sqrt(np.dot(direction, direction)))
    
        scal_accel = (G*mass2)/(distance**2)
        accel = scal_accel*unit_direction
    
        return accel



def dcoord_dt(t, coord, mass):
    """
    This gives our time dynamics. It computes the first derivative 
    with respect to time for each position and velocity variable.
    Our output is a matrix of size (#planets, 6).
     

    Parameters
    ----------
    t : float
        although t is not used for our computation, I included it to
        match the callable function signature for solve_ivp
    coord : numpy array
        coord is a flattened numpy array of size #planets*6 
        that gives us the position and velocity of each planet. 
        It should have the form [x1,y1,z1,vx1,vy1,vz1, . . . ,
        xn,yn,zn,vxn,vyn,vzn]] where n is the # of planets.
    mass: numpy array
        mass is a numpy array of size #planets that gives us the mass
        of each planet. It should have the form [m1,. . . ,mn]
        where n is the # of planets.

    Returns
    -------
    A numpy array of size #planets*6) which contains the first time 
    derivatives of every element in our input array coord.
    """
    
    #creating a copy of coord, so our initial conditions don't change
    dynamics = np.copy(coord)
    
    #computing each row of the matrix i.e. for each planet
    for planet in range(NUM_PLANETS):
        
        #computing superimposed acceleration for planet at index j
        
        #the position of planet
        position1 = coord[planet*6 : planet*6 + 3]
        #initializing dv_dt for superposition
        dv_dt = np.zeros(3)
        for j in range(NUM_PLANETS):
            
            mass2 = mass[j]
            
            position2 = coord[j*6 : j*6 + 3]
            
            #1st derivative of velocity i.e. acceleration
            #dv_dt from one planet
            sub_dv_dt = grav_accel(position1, position2, mass2)
            dv_dt = np.add(dv_dt, sub_dv_dt) #summing for superposition

        
        #replacing elements in coords with time derivatives
        velocity = coord[planet*6 + 3: planet*6 + 6]
        
        dynamics[planet*6 : planet*6 + 3] = velocity
        dynamics[planet*6 + 3 : planet*6 + 6] = dv_dt
        
    return dynamics
    
#########################################################################     
#Now we have our dynamics function
#And we have the initial conditions from our COORD array
#We can solve for the position (and velocity of each planet at some 
#future time

def simulation(T = t_f):
    """
    This function creates randomized initial conditions and the completes
    numerical integration using the solve_ivp function.
    
    Parameters
    ----------
    T: float
        T is the final time of our numerical integration
    
    Returns
    -------
    The output of solve_ivp

    """
    #seed for repeatability
    rand.seed()
    
    #randomize initial conditions
    MASS = initialize()[0]
    COORD = initialize()[1]
    
    final = solve_ivp(dcoord_dt, [0,T], COORD, args = [MASS], 
                      method = 'RK45',rtol = 1e-8, atol = 1e-8)
        
    return (final, MASS)

def plot_planets(final):
    """
    Given the output of the numerical integration function solve_ivp,
    this function will plot the position of the planets throughout the
    integration.
    
    Parameters
    ----------
    final: output of solve_ivp
    MASS: numpy array
        MASS is a numpy array of size #planets that gives us the mass
        of each planet. It should have the form [m1,. . . ,mn]
        where n is the # of planets. Returning MASS makes it accessible
        outside the function.

    """
    num_evals = np.shape(final.y)[1]

    #reshape the output (position and velocity evaluations at different 
    #timesteps) to make indexing easier
    coords = np.reshape(final.y, (NUM_PLANETS, 6, num_evals))
    
    #creating datapoints of x, y, and z coordinates 
    #for each planet (the first five if more planets are specified)
    
    x0_pos = coords[0, 0, ::]
    y0_pos = coords[0, 1, ::]
    z0_pos = coords[0, 2, ::]
    
    x1_pos = coords[1, 0, ::]
    y1_pos = coords[1, 1, ::]
    z1_pos = coords[1, 2, ::]
    
    x2_pos = coords[2, 0, ::]
    y2_pos = coords[2, 1, ::]
    z2_pos = coords[2, 2, ::]
    
    x3_pos = coords[3, 0, ::]
    y3_pos = coords[3, 1, ::]
    z3_pos = coords[3, 2, ::]
    
    x4_pos = coords[4, 0, ::]
    y4_pos = coords[4, 1, ::]
    z4_pos = coords[4, 2, ::]

    
    plt.figure()
    
    #create a 3-dimensional plot showing trajectories
    ax = plt.axes(projection = '3d')
    ax.plot3D(x0_pos, y0_pos, z0_pos, 'purple')
    ax.plot3D(x1_pos, y1_pos, z1_pos, 'cyan')
    ax.plot3D(x2_pos, y2_pos, z2_pos, 'green')
    ax.plot3D(x3_pos, y3_pos, z3_pos, 'blue')
    ax.plot3D(x4_pos, y4_pos, z4_pos, 'red')
    
    
    plt.figure()
    
    #graphs the projection onto the x-y plane
    plt.plot(x0_pos, y0_pos, 'purple')
    plt.plot(x1_pos, y1_pos, 'cyan')
    plt.plot(x2_pos, y2_pos, 'green')
    plt.plot(x3_pos, y3_pos, 'blue')
    plt.plot(x4_pos, y4_pos, 'red')
    
    plt.show()

if __name__ == "__main__":
    
    final = simulation()[0]

    print(final.message)
    
    plot_planets(final)
    



        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    