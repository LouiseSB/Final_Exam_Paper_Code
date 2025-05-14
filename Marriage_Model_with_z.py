
# import packages and set directory 

import pandas as pd # to import eg. excel
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path
from scipy import stats
import math


home = str(Path.home())
Path = home + "/Library/CloudStorage/OneDrive-Personligt/Dokumenter/Universitetet/8. Semester/Micro and Macro Models of the Labour Market/Exam Paper"
datapath = Path + "/Python"




#%% import data (distribution of income for male and female)

n_types = 50
n = 50

import_male = pd.read_csv(datapath+"/income_distribution_male.csv").to_numpy(copy=True)
# 50x2 - 50x[1] is income in currency and 50x[2] is the density 
import_female = pd.read_csv(datapath+"/income_distribution_female.csv").to_numpy(copy=True)
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
    # n = the first dimension in the matrix 
    if result == 'male':
    # if the second input is male then do the below
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
    # note that this does not return a matrix but an array (vector) 

# production function (simple)
def production_function(x,y):
    return x*y
# we use a simple production function simply x*y meaning that the two imputs we insert will be multiplied 

#Alternative production function 1 - Cobb Douglas
def production_function_Cobb_Douglas(x,y, gamma):
    return x**gamma * y**(1-gamma) 

#Alternative production function 2 - CES
def production_function_CES(x,y, gamma, rho):
    if rho == 0:
        return x**gamma * y**(1 - gamma)  # Cobb–Douglas
    else:
        return (gamma*(x**rho) + (1-gamma)*(y**rho))**(1/rho) 

#Alternative production function 3 - Similarity-based
def production_function_neg_abs_dis(x,y):
    return -abs(x - y)

# Equation (11) and (12)
def flow_update(dens_e, dens_u_o, alphas, c_delta, result, c_lambda_m, xstep): 
    int_u_o = integrate_red(dens_u_o * alphas, result, xstep) 
    # this is only the integral of u times alpha 
    # note this is always done for the opisite sex so for men we integrate u_f(y) * alpha over dy 
    # xstep is the same for both sex so we can simply use xstep as dy as well 
    # u is 1,50 (male) (opisite) and alpha is a matrix of 50x50 --> becomes a 50x50 and then interated --> 1x50 (male (opisite))
    int_u_o.shape = dens_e.shape
    # we make sure int_u_o is the same shape as e_m(x) or e_f(y) they have to be multiplied in the next line 
    # this might be because male runs the integral for women returning a array with the oppisite simentions
    return c_delta*dens_e / (c_delta + c_lambda_m * int_u_o)
    # returns the whole formula with the constants as well 
    
def chebychev_grid(n, x0, x1):
    """Returns the Chebyshev grid of size n in the interval [x0,x1]"""
    grid = np.linspace(0, n, n+1)
    nodes = (x1+x0)/2 + (x1-x0)/2 * np.cos(math.pi*grid/n)
    return np.sort(nodes)
 
def cc_weights(n):
    """
    Returns the weights used in the Clenshaw-Curtis quadrature approximation.
    n = is the size of the grid for which the weights should be created
    """
    N = np.arange(1,n,2)
    l = len(N)
    m = n-l
    v0 = 2/N/(N-2)
    v0 = np.hstack((v0, 1/N[-1]))
    v0 = np.hstack((v0, np.zeros(m)))
    index_a = np.arange(0,len(v0)-1,1)
    index_b = np.arange(len(v0)-1,0,-1)
    v2 = -v0[index_a]-v0[index_b]
    g0 = -np.ones(n)
    g0[l] = g0[l]+n
    g0[m] = g0[m]+n
    g = g0/(n**2 - 1 + n%2)
    wcc = np.fft.ifft(v2+g).real
    wcc = np.hstack((wcc, wcc[0]))
    return wcc


def integr_z(c_xy, s, s_o, gridsize, n_types, z_mean, z_sd):

    #Integrates over z, given the joint home production function C_xy,

    #the value of singlehood of the specified sex (s), the value of singlehood

    #of the other sex (s_o) and the gridsize, which should be used in approximating

    #the integral. Note that this function always uses Chebychev Grids.

    # 50x50x50
    ones = np.ones((n_types, n_types, gridsize))
    
    
    z_min = stats.norm.ppf(0.0001, loc=z_mean, scale=z_sd)
    # -3,719
    z_max = -z_min
    # 3,719

    z_grid = chebychev_grid(gridsize - 1, z_mean, z_sd)
    # (50,) from 3,719 to -3,719

    z_grid.shape = (1, 1, gridsize)
    # 1x1x50

    z_vals = ones * z_grid
    # 50x50x50

    # print(np.shape(z_vals))

    new_order = (1, 0, 2)
    # tuple 

    z_vals_prime = np.transpose(z_vals,axes=new_order)
    # 50x50x50
    # change the order so axis 1 is swaped with axis 0

    # print(np.shape(z_vals_prime))

    z_weights = cc_weights(gridsize - 1)
    # (50,) 0< weights <1
    

    #we need the 2d version of c_xy transposed

    C_xy_prime = np.transpose(c_xy)
    # 50 x 50 

    # Making the joint home production function temporarily three dimensional,
    # so I can integrate over the third dimension.

    tmp_shape_cxy = c_xy.shape
    # tuple (50,50)

    tmp_shape_cxy_prime = C_xy_prime.shape
    # tuple (50,50)

    c_xy.shape = (n_types, n_types, 1)
    # c_xy is now 50x50x1

    C_xy_prime.shape = (n_types, n_types, 1)
    # c_xy_prime is now 50x50x1

    # print(np.shape(C_xy))

    tmp_shape_s = s.shape
    # tuple (1,50)
    

    s.shape = (s.shape[0], s.shape[1], 1)
    # now s_m_1 becomes 1,50,1 

    # print(np.shape(s))

    tmp_shape_so = s_o.shape
    # tuple (50,1)

    s_o.shape = (s_o.shape[0], s_o.shape[1], 1)
    # now s_f_1 becomes 50,1,1 

    # print(np.shape(s_o))

    value = z_vals + z_vals_prime + c_xy + C_xy_prime - s_o - s # change here for new values of singlehood
    # 50x50x50
    
    # max{C(x,y) - s_o, s}

    # sp_pos = np.where(sp < s, s, sp)

    sp_pos = np.where(value > 0, value, 0)

    int_z = (z_max - z_min) / 2 * np.sum(sp_pos * stats.norm.pdf(z_vals, loc=z_mean, scale=z_sd) * z_weights, axis=2)

    # if np.min(int_z)<0:

    #     print(int_z)

    c_xy.shape = tmp_shape_cxy
    C_xy_prime.shape = tmp_shape_cxy_prime
    s.shape = tmp_shape_s
    s_o.shape = tmp_shape_so
    # shape back to the shape it was before
    
    return int_z 
    

#%%

# Define dictionary
p = dict()

# model parameters
#nash-barganing power
p['c_beta'] = 0.5
#Discount rate
p['c_r'] = 0.05
#Divorce rate
p['c_delta'] = 0.1
#Meeting probability
p['c_lambda'] = 1
#Gridsize
p['gridsize'] = 50
#z_mean
p['z_mean'] = 0
#z_sd
p['z_sd'] = 1
#Sigma
p['sigma'] = 1000
#Gamma
p['c_gamma']=0.5
#Rho
p['c_rho']=-1


print(np.shape(import_male))
p['xmin']= import_male[0,0] 
print('Lowest income grid point:', p['xmin'])
p['xmax']= import_male[49,0]
print('Highest income grid point:', p['xmax'])
p['xstep']= import_male[1,0] - import_male[0,0]
print('stepsize:',p['xstep'])

# type space
p['typespace_n'] = import_male[:,0]
p['typespace'] = p['typespace_n']/np.min(p['typespace_n'])

p['n_types']=n_types
p['male_inc_dens'] = import_male[:,1] # kolon angiver at vi tager en hel kolonne
p['female_inc_dens'] = import_female[:,1]


#normalize densities
#density function for all agents 
e_m = p['male_inc_dens'] / integrate_uni(p['male_inc_dens'],p['xstep'])
e_m.shape = (1, p['n_types'])
e_f = p['female_inc_dens'] / integrate_uni(p['female_inc_dens'],p['xstep'])
e_f.shape = (p['n_types'],1)

xgrid = p['typespace'].ravel() 
ygrid = p['typespace'].ravel()

c_x = np.zeros((1, p['n_types']))
c_y = np.zeros((p['n_types'], 1))

        
for xi in range(p['n_types']):
    c_x[0,xi]=xgrid[xi]
    
for yi in range(p['n_types']):
    c_y[yi,0]=ygrid[yi]
    

alphas = np.zeros([n_types, n_types])

surplus = np.zeros([n_types, n_types])
keep_iterating = True

# main loop 
def solve_model(n, delta, lam, r, beta, tol, text,
                c_x, c_y, xstep, sigma, prod, gamma, rho):

    grid = p['typespace']

    payoffs_m = c_x
    payoffs_f = c_y
    X, Y = np.meshgrid(grid, grid, indexing='ij')
    
    if prod == 'simple':
        payoffs_couples = production_function(X, Y)
    if prod == 'Cobb-Douglas': 
        payoffs_couples = production_function_Cobb_Douglas(X,Y, gamma)
    if prod == 'CES':
        payoffs_couples = production_function_CES(X,Y, gamma, rho)
    elif prod == 'neg_abs_dis':
        payoffs_couples = production_function_neg_abs_dis(X,Y)
        

    alphas = np.ones((n, n))
    u_m = np.repeat(0.1, n).reshape(1, n)
    u_f = np.repeat(0.1, n).reshape(n, 1)
    s_m = np.repeat(0.3, n).reshape(1, n)
    s_f = np.repeat(0.3, n).reshape(n, 1)

    keep_iterating = True
    max_outer_iterations = 500
    iteration = 0

    while keep_iterating and iteration < max_outer_iterations:
        iteration += 1

        u_m_outer_prev = u_m.copy()
        u_f_outer_prev = u_f.copy()
        s_m_outer_prev = s_m.copy()
        s_f_outer_prev = s_f.copy()
        alphas_outer_prev = alphas.copy()

        # === STEP 1: Update u_m and u_f ===
        err_u = sys.float_info.max
        u_m_inner_prev = u_m.copy()
        u_f_inner_prev = u_f.copy()

        while err_u > tol:
            u_m = flow_update(e_m, u_m, alphas, p['c_delta'], 'male', p['c_lambda'], p['xstep']).reshape(1, n)
            u_f = flow_update(e_f, u_f, alphas, p['c_delta'], 'female',p['c_lambda'], p['xstep']).reshape(n, 1)
            err_u = max(np.linalg.norm(u_m - u_m_inner_prev), np.linalg.norm(u_f - u_f_inner_prev))
            u_m_inner_prev = u_m.copy()
            u_f_inner_prev = u_f.copy()
        
            int_U_m = integrate_uni(u_m, p['xstep'])
            int_U_f = integrate_uni(u_f, p['xstep'])


        # === STEP 2: Update s_m and s_f ===
        err_s = sys.float_info.max
        s_m_inner_prev = s_m.copy()
        s_f_inner_prev = s_f.copy()

        while err_s > tol:
            expected_surplus = integr_z(payoffs_couples, s_m, s_f, p['gridsize'], n_types, p['z_mean'], p['z_sd'])
            expected_surplus = np.asarray(expected_surplus).reshape((n_types, n_types))
             
            int_sup_m = integrate_red(expected_surplus * u_f, 'male', xstep).reshape(1, n)/n
            int_sup_f = integrate_red(expected_surplus * u_m, 'female', xstep).reshape(n, 1)/n

            s_m = ((payoffs_m + (lam * beta / (r + delta)) * int_sup_m) / (1+((lam * beta / (r + delta))*int_U_f))).reshape(1,n)
            s_f = ((payoffs_f + (lam * (1 - beta) / (r + delta)) * int_sup_f) / (1+((lam * (1 - beta) / (r + delta))*int_U_m))).reshape(n,1)
            
            #print('shape_s_m', s_m.shape)

            err_s = max(np.linalg.norm(s_m - s_m_inner_prev), np.linalg.norm(s_f - s_f_inner_prev))
            s_m_inner_prev = s_m.copy()
            s_f_inner_prev = s_f.copy()

        # === STEP 3: Update matching probabilities ===
        surplus = -payoffs_couples + s_m + s_f
        alphas = 1 - stats.norm.cdf(surplus / sigma ,loc=p['z_mean'],scale=p['z_sd']) 


        # === STEP 4: Outer Convergence ===
        outer_error = max(
            np.linalg.norm(u_m - u_m_outer_prev),
            np.linalg.norm(u_f - u_f_outer_prev),
            np.linalg.norm(s_m - s_m_outer_prev),
            np.linalg.norm(s_f - s_f_outer_prev),
            np.linalg.norm(alphas - alphas_outer_prev)
        )

        if text:
            print(f"Iteration {iteration}: outer error = {outer_error:.2e}")

        if outer_error < tol:
            keep_iterating = False

    if text:
        print(f"Model converged in {iteration} outer iterations")

    return grid, u_m, u_f, s_m, s_f, alphas, int_U_m, int_U_f, payoffs_couples
    
   
#For simple production function      
grid_simple, u_m_simple, u_f_simple, s_m_simple, s_f_simple, alphas_simple, int_U_m_simple, int_U_f_simple, payoffs_couples_simple= solve_model(
    n, p['c_delta'], p['c_lambda'], p['c_r'], p['c_beta'], 1e-12, True, 
    c_x, c_y, p['xstep'], p['sigma'], 'simple', p['c_gamma'], p['c_rho']
)

#For Cobb-Douglas production function
grid_CB, u_m_CB, u_f_CB, s_m_CB, s_f_CB, alphas_CB, int_U_m_CB, int_U_f_CB, payoffs_couples_CB= solve_model(
    n, p['c_delta'], p['c_lambda'], p['c_r'], p['c_beta'], 1e-12, True, 
    c_x, c_y, p['xstep'], p['sigma'], 'Cobb-Douglas', p['c_gamma'], p['c_rho']
)

#For CES production function
grid_CES, u_m_CES, u_f_CES, s_m_CES, s_f_CES, alphas_CES, int_U_m_CES, int_U_f_CES, payoffs_couples_CES= solve_model(
    n, p['c_delta'], p['c_lambda'], p['c_r'], p['c_beta'], 1e-12, True, 
    c_x, c_y, p['xstep'], p['sigma'], 'CES', p['c_gamma'], p['c_rho']
)

#For CES production function
grid_neg, u_m_neg, u_f_neg, s_m_neg, s_f_neg, alphas_neg, int_U_m_neg, int_U_f_neg, payoffs_couples_neg= solve_model(
    n, p['c_delta'], p['c_lambda'], p['c_r'], p['c_beta'], 1e-12, True, 
    c_x, c_y, p['xstep'], p['sigma'], 'neg_abs_dis', p['c_gamma'], p['c_rho']
)

#Calculating joint density of matches
n_xy_simple = (p["c_lambda"]*u_m_simple*u_f_simple*alphas_simple)/p["c_delta"]

n_xy_CB = (p["c_lambda"]*u_m_CB*u_f_CB*alphas_CB)/p["c_delta"]

n_xy_CES = (p["c_lambda"]*u_m_CES*u_f_CES*alphas_CES)/p["c_delta"]

n_xy_neg = (p["c_lambda"]*u_m_neg*u_f_neg*alphas_neg)/p["c_delta"]

#Total surplus
s_xy_simple = payoffs_couples_simple - s_f_simple - s_m_simple

s_xy_CB = payoffs_couples_CB - s_f_CB - s_m_CB

s_xy_CES = payoffs_couples_CES - s_f_CES - s_m_CES

s_xy_neg = payoffs_couples_neg - s_f_neg - s_m_neg
    
#%% Plotting
    
#Alphas
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

fig = wireframe_with_heatmap(alphas_simple, p['typespace_n'], 250, 30, r'$\alpha(x,y)$')
fig.savefig("alpha_wireframe_and_heatmap.png")   

fig1 = wireframe_with_heatmap(alphas_CB, p['typespace_n'], 250, 30, r'$\alpha(x,y)$')
fig1.savefig("alpha_wireframe_and_heatmap_CB.png")  

fig2 = wireframe_with_heatmap(alphas_CES, p['typespace_n'], 250, 30, r'$\alpha(x,y)$')
fig2.savefig("alpha_wireframe_and_heatmap_CES.png")    
 
fig3 = wireframe_with_heatmap(alphas_neg, p['typespace_n'], 250, 30, r'$\alpha(x,y)$')
fig3.savefig("alpha_wireframe_and_heatmap_neg.png")   

# Plot of c_xy, s_xy and n_xy
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
    plt.savefig("all_wireframes_z.png")
    plt.show()

# Example usage
plot_three_wireframes(
    z1=payoffs_couples_simple,
    z2=s_xy_simple,
    z3=n_xy_simple,
    space=p['typespace_n'],
    titles=[r'$c(x,y)$', r'$s(x,y)$', r'$n(x,y)$']
)


def wireframeplot2(z, space, azim, elev, title):

    """

    Produces a simple wireframe plot.

    """

    X, Y = np.meshgrid(space, space)

    fig = plt.figure(figsize=(10,7))

    ax = fig.add_subplot(projection='3d')

    ax.plot_wireframe(X, Y, z,

                          rstride=2,

                          cstride=2,

                          color='DarkSlateBlue',

                          linewidth=1,

                          antialiased=True)

    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.set_rotate_label(False)  # disable automatic rotation

    ax.yaxis.set_rotate_label(False)  # disable automatic rotation

#    ax.zaxis.set_rotate_label(False)  # disable automatic rotation

 

#    ax.set_ylabel(r'\textbf{Women}', labelpad=20,rotation='horizontal')

#    ax.set_xlabel(r'\textbf{Men}', labelpad=10,rotation='horizontal')

    ax.set_ylabel('Women', labelpad=20,rotation='horizontal')

    ax.set_xlabel('Men', labelpad=10,rotation='horizontal')

 

#    ax.yaxis.set_label_coords(-5,0)

#    ax.set_zlabel(r zlabel, rotation=0)      

#    plt.xlabel(rotation='horizontal')

#    ax.set_zlabel(r'\textbf{zlabel}')

    ax.dist = 10

    plt.title(title)

    plt.show()

    return fig

#Plot c_xy
fig4 = wireframeplot2(payoffs_couples_CB, p['typespace_n'],250,30,r'c(x,y)')

fig4.savefig("c_xy_CB.png")

fig5 = wireframeplot2(payoffs_couples_CES, p['typespace_n'],250,30,r'c(x,y)')

fig5.savefig("c_xy_CES.png")

fig6 = wireframeplot2(payoffs_couples_neg, p['typespace_n'],250,30,r'c(x,y)')

fig6.savefig("c_xy_neg.png")

