#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import packages and set directory 

import pandas as pd # to import eg. excel
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D



home = str(Path.home())
Path = home + "/Documents/GitHub/MMLM"
datapath = Path + "/Data:code"


#%% import data (distribution of income for male and female)

n_types = 80

import_male = pd.read_excel(datapath+"/data_educ_age.xlsx").to_numpy(copy=True)
# 50x2 - 50x[1] is income in currency and 50x[2] is the density 
import_female = pd.read_excel(datapath+"/data_educ_age.xlsx").to_numpy(copy=True)
# same just for women meaning that 50x[1] is exactly the same as for male 


#%% Define functions 

# Checking whether we integrate to 1
def integrate_uni(values, xstep):
    "integrates the grid with the specified values with stepsize xstep"
    #spacing = x1/values.size
    copy = np.copy(values)
    # make a copy of the array 
    copy.shape = (1, values.size)
    # reshape the copied array as a 1x array size 
    return integrate.simpson(copy, dx=xstep)


def integrate_red(matrix, result, xstep): #integrating in 2 dimensions
    n = matrix.shape[0]
    # n = the first dimention in the matrix 
    if result == 'male':
    # if the second input is male then do teh below
        inner = np.zeros((1,n))
        # set inner equal to a vector (1,n) of zeroes
        for i in range(0,n):
        # for each entrance 0,n
            inner[0,i] = np.squeeze(integrate_uni(matrix[:,i],xstep))
            # np.squueze 
    elif result == 'female':
        inner = np.zeros((n,1))
        for i in range(0,n):
            inner[i,0] = np.squeeze(integrate_uni(matrix[i,:],xstep))
    return inner
    # note that this does not return a matrix but a array (vector) 

# production function (simple)
def production_function(x,y):
    return x*y
# we use a simple production function simply x*y meaning that the two imputs we insert will be multiplied 

# Equation (11) and (12)
def flow_update(dens_e, dens_u_o, alphas, c_delta, result, c_lambda, xstep): 
    int_u_o = integrate_red(dens_u_o * alphas, result, xstep) 
    # this is only the integral of u times alpha 
    # note this is always done for the opisite sex so for men we integrate u_f(y) * alpha over dy 
    # xstep is the same for both sex so we can simply use xstep as dy as well 
    # u is 1,50 (male) (opisite) and alpha is a matrix of 50x50 --> becomes a 50x50 and then interated --> 1x50 (male (opisite))
    int_u_o.shape = dens_e.shape
    # we make sure int_u_o is the same shape as e_m(x) or e_f(y) they have to be multiplied in the next line 
    # this might be because male runs the integral for women returning a array with the oppisite simentions
    return c_delta*dens_e / (c_delta + c_lambda * int_u_o)
    # returns the whole formula with the constants as well 

# Equation (13) and (14)
def flow_surplus(c, c_lambda, beta, r, delta, U, c_xy, s_o, s, u_o, result, xstep):
    if result == 'male':
        constant = ((c_lambda*beta)/(r+delta))
    elif result == 'female':
        constant = ((c_lambda*(1-beta))/(r+delta))
    denominator = 1 + constant* U
    left = c_xy - s_o
    max_1 = np.maximum(left, s)
    inner_integrand_1 = max_1 * u_o
    flow_surplus_1 = integrate_red(inner_integrand_1, result, xstep)
    norminator = c + constant * flow_surplus_1
    return norminator / denominator
    
  
#%%

# Define dictionary
p = dict()

# model parameters
# Nash-barganing power
p['c_beta'] = 0.5
# Discount rate
p['c_r']= 0.05
# Divorce rate 
p['c_delta']=0.1
# Meeting probability 
p['c_lambda']=1


print(np.shape(import_male))
p['xmin']= import_male[0,0]
print('Lowest income grid point:', p['xmin'])
p['xmax']= import_male[79,0]
print('Highest income grid point:', p['xmax'])
p['xstep']= import_male[1,0] - import_male[0,0]
print('stepsize:',p['xstep'])

# type space
p['typespace_n'] = import_male[:,0]
p['typespace'] = p['typespace_n']/np.min(p['typespace_n'])

p['n_types']=n_types
p['male_inc_dens'] = import_male[:,1] 
p['female_inc_dens'] = import_female[:,2]


#normalize densities
#density function for all agents 
e_m = p['male_inc_dens'] / integrate_uni(p['male_inc_dens'],p['xstep'])
e_m.shape = (1, p['n_types'])
e_f = p['female_inc_dens'] / integrate_uni(p['female_inc_dens'],p['xstep'])
e_f.shape = (p['n_types'],1)

xgrid = p['typespace'].ravel() 
ygrid = p['typespace'].ravel()

# initializing c_xy 
c_xy = np.zeros((p['n_types'],p['n_types']))

# flow utilities for couples
for xi in range(p['n_types']):
    for yi in range (p['n_types']):
        #absolute advantage as in shimer/smith
        c_xy[xi,yi]=production_function(p['typespace'][xi], p['typespace'][yi])
        
c_x = np.zeros((1, p['n_types']))
c_y = np.zeros((p['n_types'], 1))
        
        
for xi in range(p['n_types']):
    c_x[0,xi]=xgrid[xi]
    
for yi in range(p['n_types']):
    c_y[yi,0]=ygrid[yi]

values_s_m = np.zeros((n_types,n_types))
values_s_f = np.zeros((n_types,n_types))

alphas = np.ones([n_types, n_types])

u_m_1 = np.ones((1, p['n_types']))
u_f_1 = np.ones((p['n_types'], 1))

keep_iterating = True

#main loop
while keep_iterating:
    e = sys.float_info.max
    u_m_prev = u_m_1
    u_f_prev = u_f_1
    while e > 1e-12:
        
        
        u_m_1 = flow_update(e_m, u_m_prev, alphas, p['c_delta'], 'male', p['c_lambda'], p['xstep'])
        u_f_1 = flow_update(e_f, u_f_prev, alphas, p['c_delta'], 'female',p['c_lambda'], p['xstep'])
 
 
        e = max(
        np.linalg.norm(u_m_prev - u_m_1),
        np.linalg.norm(u_f_prev - u_f_1)
    )
        
        u_m_prev = u_m_1
        u_f_prev = u_f_1
        
    int_U_m = integrate_uni(u_m_1, p['xstep'])
    int_U_f = integrate_uni(u_f_1, p['xstep'])
    
    int_U_m_p = int_U_m
    int_U_f_p = int_U_f

    # Equation 18
    s_m_1 = flow_surplus(c_x, p['c_lambda'], p['c_beta'], p['c_r'], p['c_delta'], int_U_f_p, c_xy, values_s_f, values_s_m, u_f_prev, 'male', p['xstep'])
    s_f_1 = flow_surplus(c_y, p['c_lambda'], p['c_beta'], p['c_r'], p['c_delta'], int_U_m_p, c_xy, values_s_m, values_s_f, u_m_prev, 'female', p['xstep'])
    
    values_s_m = s_m_1
    values_s_f = s_f_1
    
     
    matrix = values_s_m + values_s_f
    
    new_alphas = np.zeros([n_types, n_types])
    for i in range(n_types):
        for j in range(n_types):
            if c_xy[i,j] - matrix[i,j] > 0:
                    new_alphas[i,j] = 1
                    
    print (n_types**2 - (new_alphas == alphas).sum())
    
    if (new_alphas == alphas).all():
            is_convergence = True
            keep_iterating = False
    else:
            alphas = new_alphas


#Calculating joint density of matches
n_xy = (p["c_lambda"]*u_m_1*u_f_1*alphas)/p["c_delta"]

# total surplus of marrige 
s_xy = c_xy - values_s_f - values_s_m

    
#%% Plotting

def wireframe_with_heatmap(z, space, azim, elev, title):
    """
    Displays a 3D wireframe plot next to a 2D heatmap.
    """
    X, Y = np.meshgrid(space, space)

    fig = plt.figure(figsize=(14, 6))

    # Wireframe subplot (left)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_wireframe(X, Y, z,
                       rstride=2, cstride=2,
                       color='DarkSlateBlue',
                       linewidth=1, antialiased=True)
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_ylabel('Women', labelpad=20, rotation='horizontal')
    ax1.set_xlabel('Men', labelpad=10, rotation='horizontal')
    ax1.set_title(title)

    # Heatmap subplot (right)
    ax2 = fig.add_subplot(1, 2, 2)
    heatmap = ax2.imshow(z, origin='lower', extent=[space[0], space[-1], space[0], space[-1]],
                         aspect='auto', cmap='viridis')
    ax2.set_xlabel('Men')
    ax2.set_ylabel('Women')
    ax2.set_title('Heatmap of ' + title)
    plt.colorbar(heatmap, ax=ax2, orientation='vertical')

    plt.tight_layout()
    plt.show()

    return fig

# Example usage
fig = wireframe_with_heatmap(alphas, p['typespace_n'], 250, 30, r'$\alpha(x,y)$')
fig.savefig("alpha_wireframe_and_heatmap.png")

# plot of all three (c_xy, s_xy, n_xy)
def plot_three_wireframes(z1, z2, z3, space, titles, elev=30, azim=250):
    """
    Plots 3 wireframe plots in a 2x2 layout:
    [ z1 | z2 ]
    [   z3    ]
    """

    X, Y = np.meshgrid(space, space)

    fig = plt.figure(figsize=(14, 10))

    # Use gridspec to arrange subplots
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax3 = fig.add_subplot(gs[1, :], projection='3d')  # span both columns

    axes = [ax1, ax2, ax3]
    zs = [z1, z2, z3]

    for ax, z, title in zip(axes, zs, titles):
        ax.plot_wireframe(X, Y, z,
                          rstride=2,
                          cstride=2,
                          color='DarkSlateBlue',
                          linewidth=1,
                          antialiased=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('Men', labelpad=10)
        ax.set_ylabel('Women', labelpad=20)
        ax.set_title(title)
        ax.dist = 10

    plt.tight_layout()
    plt.savefig("all_wireframes.png")
    plt.show()

# Example usage
plot_three_wireframes(
    z1=c_xy,
    z2=s_xy,
    z3=n_xy,
    space=p['typespace_n'],
    titles=[r'$c(x,y)$', r'$s(x,y)$', r'$n(x,y)$']
)



x = np.linspace(0, 80, num=80)

# Assuming e_f and e_m are arrays of length 79
plt.plot(x, e_f, label='e_f (female type)')
plt.plot(x, np.transpose(e_m), label='e_m (male type)')

plt.xlabel("Type")
plt.ylabel("Effort")
plt.title("Effort by Type for Females and Males")
plt.grid(True)
plt.legend()
plt.show()



