import pandas as pd
import numpy as np
import scipy as sp
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import requests, zipfile, io # So that we can download and unzip files\n",
from income_process import income_process
#from matplotlib2tikz import save as tikz_save
import matplotlib
import statsmodels.tsa.filters as filters
import quandl as Quandl
import quantecon as qe
import seaborn as sns
#import warnings; warnings.simplefilter('ignore')

from poly_base import interpolate as ip
from scipy import sparse
# Parameters:
Beta = 0.97
tau = 5.0
sigma = -2. # IMPORTANT! Satisfy the Gross Substitution property 
gamma =  2 # CRRA utility 
dampen = 1 # dampening parameter
delta = 0.05
max_tol = 1e-10 # Tolerance in the maximization
val_tol = 1e-9 # Tolerance in the value function iteration
stat_tol = 1e-8 # Tolerance in the stationary distribution
root_tol = 1e-1 # Tolerance in the root finding
#Skill_High = 1.5
#Skill_Med = 1
#Skill_Low = 0.75
#Skill_level = np.array([Skill_High,Skill_Med,Skill_Low])
#Skill_level_no = Skill_level.shape[0]
#Skill_level1 = np.zeros((3,2))
#Skill_level1[:,0]=Skill_level
#Skill_High_Foreign = 1.6
#Skill_Med_Foreign = 1.2
#Skill_Low_Foreign = 0.9
#Skill_level_Foreign =  np.array([Skill_High_Foreign,Skill_Med_Foreign,Skill_Low_Foreign])
#Skill_level1[:,1]=Skill_level_Foreign
#Domestic_Transition_Up = 0.12
#Domestic_Transition_Down = 0.1
#Foreign_Transition_Up = 0.15
#Foreign_Transition_Down = 0.1
citizenship_attrition = 0#0.005 # half a percent chance of becoming local citizen
No_countries = 2
freedom_fairy = 0.5
#freedom_fairy1=[0.5,1]
remmittance_savings_rate = 0.00#.5 #proportion of your assets held in the location of your citizenship 1
TFP_Domestic = 0.865
TFP_Foreign = 1.0
fastcoeff = 2 # Number of Bellman iterations
alpha = np.array([0.43,0.33])#([0.432,0.33])
alpha_l = np.array([0.5,0.50])#([0.498 ,0.6]) # elasticity on labor in the production function
Total_Labor_Domestic = 0.35
Total_Labor_Foreign = 0.65
degree = 1 # spline approximation - linear
degree_fine = 1# distribution approximation
TFP = np.array([TFP_Domestic] + [TFP_Foreign])
Czship = np.array([Total_Labor_Domestic] + [Total_Labor_Foreign])
#TFP = np.array([TFP_Domestic] + [TFP_Foreign])
#Czship = np.array([Total_Labor_Domestic] + [Total_Labor_Foreign]+ [Total_Labor_Foreign])
#home_pref = 0.08*np.ones((No_countries,1))
#home_pref_repeated = np.reshape(np.repeat(home_pref,No_countries*Skill_level_no),(No_countries**2*Skill_level_no,1))


#Skill_Matrix_Domestic = np.zeros((Skill_level_no,Skill_level_no))
#Skill_Matrix_Domestic[0,0]=1-Domestic_Transition_Down
#Skill_Matrix_Domestic[0,1]=Domestic_Transition_Down
#Skill_Matrix_Domestic[1,0]=Domestic_Transition_Up
#Skill_Matrix_Domestic[1,1]=1-Domestic_Transition_Up-Domestic_Transition_Down
#Skill_Matrix_Domestic[1,2]=Domestic_Transition_Down
#Skill_Matrix_Domestic[2,1]=Domestic_Transition_Up
#Skill_Matrix_Domestic[2,2]=1-Domestic_Transition_Up
#
#
#Skill_Matrix_Foreign = np.zeros((Skill_level_no,Skill_level_no))
#Skill_Matrix_Foreign[0,0]=1-Foreign_Transition_Down
#Skill_Matrix_Foreign[0,1]=Foreign_Transition_Down
#Skill_Matrix_Foreign[1,0]=Foreign_Transition_Up
#Skill_Matrix_Foreign[1,1]=1-Foreign_Transition_Up-Foreign_Transition_Down
#Skill_Matrix_Foreign[1,2]=Foreign_Transition_Down
#Skill_Matrix_Foreign[2,1]=Foreign_Transition_Up
#Skill_Matrix_Foreign[2,2]=1-Foreign_Transition_Up
## Multiple regions - concatenate with for loop each skill matrix!
#Skill_Matrix_stacked =  np.concatenate((Skill_Matrix_Domestic,Skill_Matrix_Foreign),axis =1)

Skill_level1,Skill_Matrix_stacked,Skill_level_no,Transition_Matrix = income_process()  #,prod_norm_level
Skill_Matrix_stacked1 = np.tile(Skill_Matrix_stacked,(No_countries,1))
Skill_level_stacked = Skill_level1.T.flatten().reshape((No_countries*Skill_level_no,1))

param_asset = (50,0.0,40.0)
curv_a = 0.15
asset_grid = ip.wealth_knot(param_asset, degree=1, curv=curv_a)
#asset_grid = np.linspace(param_asset[1],param_asset[2],param_asset[0])
fine_grid_no = 200
asset_grid_fine =  ip.equidistant_nonlin_grid(asset_grid,fine_grid_no) #np.linspace(param_asset[1],param_asset[2],fine_grid_no)
#asset_grid_fine = ip.wealth_knot((fine_grid_no,param_asset[1],param_asset[2]))
asset_grid1 = asset_grid.reshape((param_asset[0],1))
asset_grid_fine1 = asset_grid_fine.reshape((fine_grid_no,1))
knots_a = asset_grid #ip.wealth_knot(param_asset, degree=degree, curv=0.15)
#knots_a_fine = ip.wealth_knot((fine_grid_no,param_asset[1],param_asset[2]), degree=degree_fine, curv=0.15)
#loc selector matrix:
loc_selector = np.kron(np.ones((1,No_countries)),np.kron(np.eye(No_countries,No_countries),np.ones((1,Skill_level_no*fine_grid_no)))) 
# citizenship selector matrix:
cit_selector = np.kron(np.kron(np.eye(No_countries,No_countries),np.ones((1,Skill_level_no*fine_grid_no))),np.ones((1,No_countries)))

asset_grid_fine1_tile = np.tile(asset_grid_fine1,(Skill_level_no * No_countries**2,1))
#skill_level_tile = np.tile(np.repeat(Skill_level,fine_grid_no),(1,No_countries**2)).T
#Skill_level_stacked = np.concatenate((Skill_level,Skill_level_Foreign),axis =0)

skill_level_tile = np.tile(np.repeat(Skill_level_stacked,fine_grid_no),(1,No_countries)).T

init_distr = (np.kron(np.eye(No_countries).reshape((No_countries**2,1)),np.ones((  Skill_level_no * fine_grid_no,1))) / (No_countries
        * Skill_level_no * fine_grid_no)) # Initially all citizens live in their location. Only has an effect if there is absorbing state

Transition_Matrix1 = np.kron(np.eye(No_countries),Transition_Matrix[:Skill_level_no,:Skill_level_no]) + np.kron(
    (np.ones((No_countries,No_countries)) - np.eye(No_countries)),Transition_Matrix[Skill_level_no:,:Skill_level_no] )
#Skill_Matrix_stacked =  np.concatenate((Skill_Matrix_Domestic,Skill_Matrix_Foreign),axis =1)

Transition_Matrix_stacked = np.zeros((No_countries * Skill_level_no,No_countries*Skill_level_no))
#Transition_Matrix_stacked1 = np.zeros((No_countries**2 * Skill_level_no,No_countries**2 *Skill_level_no,No_countries))
#Transition_Matrix_stacked_stay = np.zeros((No_countries**2 * Skill_level_no,No_countries**2 *Skill_level_no))
Util_cost_matrix = tau * np.ones((No_countries**2 * Skill_level_no,param_asset[0],No_countries))
Util_cost_matrix_fine = tau * np.ones((No_countries**2 * Skill_level_no,fine_grid_no,No_countries))

for co in range(No_countries):
    for co2 in range(No_countries):
        Transition_Matrix_stacked[(co*Skill_level_no):((co + 1)*Skill_level_no),(co2*Skill_level_no):((co2 + 1)*Skill_level_no)
            ] = Transition_Matrix1[(co*Skill_level_no):((co + 1)*Skill_level_no),(co2*Skill_level_no):((co2 + 1)*Skill_level_no)
            ] @ Skill_Matrix_stacked1[(co*Skill_level_no):((co + 1)*Skill_level_no),(
            co2*Skill_level_no):((co2 + 1)*Skill_level_no)
            ]
for co in range(No_countries):
    index_tod = co*Skill_level_no
    index_tod1 = (co + 1)*Skill_level_no
    # co is current location 
    for co2 in range(No_countries):
        #co2 is current citizenship 
        index_tod2 = co2 * (No_countries * Skill_level_no)
        index_tod3 = (co2+1) * (No_countries * Skill_level_no)
#        print(index_tod,index_tod1,index_tod2,index_tod3)
#        
#        Transition_Matrix_stacked1[index_tod:index_tod1]
#        Transition_Matrix_stacked1[index_tod2:index_tod3,(index_tod2 + index_tod):(index_tod2 +index_tod1),co
#                                  ] = Transition_Matrix_stacked[:,index_tod:index_tod1]
#        Transition_Matrix_stacked_stay[(index_tod + index_tod2
#                                        ):(index_tod1 + index_tod2),(index_tod + index_tod2
#                                        ):(index_tod1 + index_tod2)] = Transition_Matrix_stacked[
#            index_tod:index_tod1,index_tod:index_tod1] 
        Util_cost_matrix[(index_tod + index_tod2
                                        ):(index_tod1 + index_tod2),:,co] = 0
        Util_cost_matrix_fine[(index_tod + index_tod2
                                        ):(index_tod1 + index_tod2),:,co] = 0
    
    
Util_cost_matrix[6:12,:,0]=0
Util_cost_matrix[12:18,:,1]=0

Util_cost_matrix_fine[6:12,:,0]=0
Util_cost_matrix_fine[12:18,:,1]=0
#Transition_Matrix_stacked1[3:6,0:3,0]=Transition_Matrix_stacked[0:3,0:3]
#Transition_Matrix_stacked1[6:9,9:12,1] = Transition_Matrix_stacked[3:6,3:6]
Transition_Matrix_stacked1 = np.zeros((No_countries**2 * Skill_level_no,No_countries**2 *Skill_level_no,No_countries))
Transition_Matrix_stacked_stay = np.zeros((No_countries**2 * Skill_level_no,No_countries**2 *Skill_level_no))
for co2 in range(No_countries):
    # citizenship
    for co1 in range(No_countries):
        #current location
        for skill_it in range(Skill_level_no):
            #skill
            index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
            for co3 in range(No_countries):
                #Choice of location
                for co4 in range(No_countries):
                    #citizenship transition
                    for skill_it2 in range(Skill_level_no):
                        #skill transition
                        index_tod1 = skill_it2 + co3* Skill_level_no + co4 * No_countries * Skill_level_no
                        if(co3 == co1): # current location is the same as before
                            if(co4==co2 and co3 ==co2):# citizenship is unchanged and equals the location # co1=co2=co3=co4
                                Transition_Matrix_stacked1[index_tod,index_tod1,co3] = Transition_Matrix_stacked[co1*Skill_level_no + skill_it,co1*Skill_level_no + skill_it2]
                                Transition_Matrix_stacked_stay[index_tod,index_tod1] = Transition_Matrix_stacked[co1*Skill_level_no + skill_it,co1*Skill_level_no + skill_it2]
                            if(co4==co2 and co3 !=co2):# citizenship is unchanged and does not equals the location #1 - citizenship attrition
                                Transition_Matrix_stacked1[index_tod,index_tod1,co3] = (1 - citizenship_attrition) *Transition_Matrix_stacked[co1*Skill_level_no + skill_it,co1*Skill_level_no + skill_it2]
                                Transition_Matrix_stacked_stay[index_tod,index_tod1] =  (1 - citizenship_attrition) *Transition_Matrix_stacked[co1*Skill_level_no + skill_it,co1*Skill_level_no + skill_it2]
                            if(co4!=co2 and co3 !=co2):# citizenship has changed as it was not equal to the location # citizenship attrition
                                Transition_Matrix_stacked1[index_tod,index_tod1,co3] = citizenship_attrition *Transition_Matrix_stacked[co1*Skill_level_no + skill_it,co1*Skill_level_no + skill_it2]
                                Transition_Matrix_stacked_stay[index_tod,index_tod1] = citizenship_attrition *Transition_Matrix_stacked[co1*Skill_level_no + skill_it,co1*Skill_level_no + skill_it2]
                        if(co3 != co1): # migration - no citizenship transition possible - still citizenship matters
                            if(co4==co2 and co3 ==co2):# citizenship is unchanged and equals the location #- returning home to co3
                                Transition_Matrix_stacked1[index_tod,index_tod1,co3] = Transition_Matrix_stacked[co3*Skill_level_no + skill_it,co3*Skill_level_no + skill_it2]
                            if(co4==co2 and co3 !=co2):# citizenship is unchanged and does not equal the location # foreingers going to co3
                                Transition_Matrix_stacked1[index_tod,index_tod1,co3] = Transition_Matrix_stacked[co4*Skill_level_no + skill_it,co3*Skill_level_no + skill_it2]
#Phi = ip.spli_basex(param_asset,asset_grid,deg = degree)
Phi = ip.spli_basex(param_asset,asset_grid,deg = degree,knots = knots_a)
#Phi_fine = ip.spli_basex(param_asset,asset_grid_fine,deg = degree)
Phi_fine = ip.spli_basex(param_asset,asset_grid_fine,deg = degree,knots = knots_a)
Phi_inv = np.linalg.inv(Phi)
Phi_inv_tile = np.tile(Phi_inv, (1,No_countries**2 * Skill_level_no))
diag_transf_v = np.kron(np.ones((No_countries**2 * Skill_level_no,1)),np.eye(param_asset[0]))
def diag_mat_block(Mat):
    res = np.zeros((Mat.shape[0] * Mat.shape[1],Mat.shape[1]))
    for ii in range(Mat.shape[1]):
        res[(ii*Mat.shape[0]):((ii + 1)*Mat.shape[0]),ii] = Mat[:,ii]
    return res
lower_bound = 0.0 * np.ones((param_asset[0],1))
lower_bound_fine = 0.0 * np.ones((fine_grid_no,1))
upper_bound = param_asset[2] * np.ones((param_asset[0],1))
upper_bound_fine = param_asset[2] * np.ones((fine_grid_no,1))

def Perturb_Q(A):
    n_ds = A.shape[0]
    #Perturb Q
    eta =( A[A>0].min() )/(2.*n_ds)
    A = np.where(A<=0.0, A + eta , A)
    #Normalize Q
    for i in range(n_ds):
        A[i,:] = A[i,:]/(np.sum(A[i,:]))
    return A

def stat_distr(Mat,res = init_distr,iterate = 400, tol = stat_tol):
    #if res.any() == None:
    #    res = np.ones((Mat.shape[0],1))/Mat.shape[0]
    res_prev = res.copy()
    conv = 10
    while conv > tol:
        res = np.linalg.matrix_power(Mat.T,iterate) @ res_prev
        conv = np.max(np.absolute(res - res_prev))
        res_prev = res.copy()
        #print(conv)
    return res
def stat_distr_eigen(Mat):
    #if res.any() == None:
    #    res = np.ones((Mat.shape[0],1))/Mat.shape[0]
    Eigenval, Eigenvec = sp.linalg.eig(Mat, left = True, right = False )
    Eigenvec=Eigenvec.real
    res = 0
    arg = np.argsort(np.abs(Eigenval - 1.))[:No_countries]
    for i in range(No_countries):
        stationary_dist =  Eigenvec[:,arg[i]]
        stationary_dist =  Czship[i] * (stationary_dist.real)/(stationary_dist.real.sum())
        res = res + stationary_dist
    return res
def stat_distr_eigen_sep(Mat):
    #if res.any() == None:
    #    res = np.ones((Mat.shape[0],1))/Mat.shape[0]
    stationary_dist1 = np.zeros((No_countries**2*Skill_level_no*fine_grid_no,1))

    for i in range(No_countries):
        Eigenval, Eigenvec = sp.linalg.eig(Mat[:,:,i], left = True, right = False )
        Eigenvec=Eigenvec.real
        arg = np.argsort(np.abs(Eigenval - 1.))[:No_countries]
        # Check if no migration equilibrium and pick to correct stationary distr:
        if np.abs(Eigenval[arg[1]] - 1.) > stat_tol:
            no_migration = 0
        else:
            no_migration = 1
        if(no_migration == 1):
            res = 0
            for ii in range(No_countries):
                stationary_dist =  Eigenvec[:,arg[ii]].real
                #stationary_dist =  Czship[ii] * (stationary_dist.real)/(stationary_dist.real.sum())
                res = res + stationary_dist
            stationary_dist = np.zeros((No_countries*Skill_level_no*fine_grid_no))
            res1 = res[i*Skill_level_no*fine_grid_no:(i+1)*Skill_level_no*fine_grid_no]
            stationary_dist[i*Skill_level_no*fine_grid_no:(i+1)*Skill_level_no*fine_grid_no] = Czship[i] * res1/res1.sum()
        else:                
            arg1 = arg[0]
            stationary_dist = Eigenvec[:,arg1]
            stationary_dist =  (stationary_dist.real)/(stationary_dist.real.sum())*Czship[i]
        stationary_dist1[i*No_countries*Skill_level_no*fine_grid_no:((i+1)*No_countries*Skill_level_no*fine_grid_no),0] = stationary_dist
    return stationary_dist1
def stat_distr_eigen_sparse(Mat):
    #if res.any() == None:
    #    res = np.ones((Mat.shape[0],1))/Mat.shape[0]
    stationary_dist1 = np.zeros((No_countries**2*Skill_level_no*fine_grid_no,1))

    for i in range(No_countries):
        Eigenval, Eigenvec = sparse.linalg.eigs(Mat[:,:,i].T,3,sigma=1)
        Eigenvec=Eigenvec.real
        sorted_eigenvals = np.argsort(np.abs(Eigenval - 1.))
        # Check if no migration equilibrium and pick to correct stationary distr:
        if np.abs(Eigenval[sorted_eigenvals[1]] - 1) > stat_tol:
            no_migration = 0
        else:
            if(np.abs(Eigenval[sorted_eigenvals[2]] - 1) < stat_tol):
                print('More than two eigenvectors found - not enough gridpoints')
                no_migration = 1
            else:
                no_migration = 1

        if(no_migration == 1):
            res = 0
            print('no_migration')
            for ii in range(No_countries):
                stationary_dist =  Eigenvec[:,sorted_eigenvals[ii]]
                #stationary_dist =  Czship[ii] * (stationary_dist.real)/(stationary_dist.real.sum())
                res = res + stationary_dist
            stationary_dist = np.zeros((No_countries*Skill_level_no*fine_grid_no))
            res1 = res[i*Skill_level_no*fine_grid_no:(i+1)*Skill_level_no*fine_grid_no]
            stationary_dist[i*Skill_level_no*fine_grid_no:(i+1)*Skill_level_no*fine_grid_no] = Czship[i] * res1/res1.sum()            
        else:
            print('Migration')                
            #arg1 = arg[0]
            stationary_dist = Eigenvec[:,sorted_eigenvals[0]]
            stationary_dist =  (stationary_dist.real)/(stationary_dist.real.sum())*Czship[i]
        stationary_dist1[i*No_countries*Skill_level_no*fine_grid_no:((i+1)*No_countries*Skill_level_no*fine_grid_no),0] = stationary_dist
    return stationary_dist1
def util(cons):
    return cons**(1 - gamma) / (1 - gamma)

def current_util(fut_asset,income,coeff1,rel_price):
    #fut_asset = rel_price * fut_asset
    #Phi_asset = ip.spli_basex(param_asset,fut_asset.flatten(),knots = knots_a)
    Phi_asset = ip.spli_basex(param_asset,fut_asset.flatten(),deg = degree,knots = knots_a)
    V_fut = (Phi_asset @ coeff1).reshape(fut_asset.shape)
    return util(income - rel_price * fut_asset) + Beta * V_fut

def Bellman_iter2(coeff,dampen,V_bar):
    coeff_next = Phi_inv_tile @ diag_mat_block(V_bar)
    conv = np.max(np.absolute(coeff_next - coeff))
    #print(np.mean(np.absolute(coeff_next - coeff),1))
    return dampen * coeff_next + (1 - dampen) * coeff ,conv

def Newton_iter2(coeff,dampen,V_bar,index_max,Phi,Phi_prime_store):
    coeff_next = coeff.copy()
    g = np.zeros((coeff.shape[0] * coeff.shape[1],1))
    D = np.zeros((g.shape[0],g.shape[0]))
    g[:,0] = np.kron(np.eye(No_countries**2 * Skill_level_no),Phi) @ coeff.flatten(order='F') - V_bar.flatten(order='F')
    loc_kron = np.kron(index_max.flatten(order='F').reshape((No_countries**2 * Skill_level_no * param_asset[0],1)),np.ones((1,No_countries**2 * Skill_level_no * param_asset[0])))
    Phi_prime_kron = D.copy()
    Phi_prime_diag = D.copy()
    for ii in range(No_countries**2 * Skill_level_no):
        MatPhi_prime = Phi_prime_store[:,:,ii]
        Phi_prime_kron[:,(ii * param_asset[0]):((ii + 1) * param_asset[0]) ] = np.tile(MatPhi_prime
                                                                                       ,(No_countries**2 * Skill_level_no,1))
        Phi_prime_diag[(ii * param_asset[0]):((ii + 1) * param_asset[0]),(ii * param_asset[0]):((ii + 1) * param_asset[0])
                      ] = MatPhi_prime
        
    for co in range(No_countries):
        D = D  + (loc_kron == co * np.ones(loc_kron.shape)) * np.kron(
            Transition_Matrix_stacked1[:,:,co],np.ones((param_asset[0],param_asset[0]))) *Beta * Phi_prime_kron
    D = freedom_fairy * D + (1 - freedom_fairy) *np.kron(
    Transition_Matrix_stacked_stay[:,:],np.ones((param_asset[0],param_asset[0]))) * Beta *Phi_prime_kron
    D = np.kron(np.eye(No_countries**2 * Skill_level_no),Phi) - D
    #improvement =  np.linalg.inv(D) @ g
    improvement = np.linalg.solve(D, g)
    for co2 in range(No_countries):
        for co1 in range(No_countries):
            for skill_it in range(Skill_level_no):
                index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
                coeff_next[:,index_tod] = coeff[:,index_tod] - improvement[(index_tod *param_asset[0]) : ((index_tod + 1) *param_asset[0]),0]
    conv = np.max(np.absolute(coeff_next - coeff))
    return dampen * coeff_next + (1 - dampen) * coeff ,conv


def stst_resid2(xsol):
    #m1 = np.max(xsol[:No_countries])
    #m2 = np.max(xsol[No_countries:])
    log_Wage = xsol[:No_countries]
    #Wage = np.exp(log_Wage - m1 - np.log(np.exp(-m1) + np.exp(log_Wage - m1)))* 3  * TFP**2 + min_Wage 
    Wage = np.exp(log_Wage)
    #Wage = log_Wage / (1 + log_Wage)* param_asset[2] * (1 - Beta) 
    log_r = xsol[No_countries:(2 * No_countries)]
    #r = np.exp(log_r - m2 - np.log(np.exp(-m2) + np.exp(log_r - m2)))  * (1/Beta - 1) 
    r = np.exp(log_r)/ (1 + np.exp(log_r)) * (1/Beta - 1)
    residual = np.zeros(xsol.shape)
    price = np.ones((No_countries))
    log_price = xsol[(2 * No_countries):]
    price[1:] = np.exp(log_price)
    
    
    # Aggregates - DRS
    drs_factor = (1 - alpha - alpha_l) 
    drs_const_k = alpha **(((1 - alpha_l) / drs_factor) ) * alpha_l **((( alpha_l) / drs_factor) ) 
    drs_const_l = alpha **((( alpha) / drs_factor) ) * alpha_l **(((1 - alpha) / drs_factor) ) 
    K = (TFP **(1 / drs_factor) * drs_const_k * (r + delta) ** ((alpha_l - 1)/drs_factor) *
         Wage  ** ((-alpha_l)/drs_factor) * price ** (1/drs_factor))
    L = (TFP **(1 / drs_factor) * drs_const_l * (r + delta) ** ((-alpha)/drs_factor) *
     Wage  ** ((alpha-1)/drs_factor) * price ** (1/drs_factor))
    production = TFP * K ** alpha * L **alpha_l
    Profits = price * production- Wage * L - (r + delta) * K # nominal profits

    
    # Workers problem
    coeff_e = 0* np.ones((param_asset[0],No_countries**2 * Skill_level_no))
    V_val = coeff_e.copy()
    V_bar_prev = np.zeros((param_asset[0],No_countries**2 * Skill_level_no))
    Phi_prime_store = np.zeros((param_asset[0],param_asset[0],No_countries**2 * Skill_level_no))
    #Value function approximation
    conv = 10
    iterate = 0
    while (conv>val_tol):
        for co2 in range(No_countries):
            for co1 in range(No_countries):
                for skill_it in range(Skill_level_no):
                    index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
                    real_income = (Wage[co1] * Skill_level1[skill_it,co1] + 
                           price[co2] * (1 + r[co2]) * asset_grid1 * remmittance_savings_rate
                           + 
                           price[co1] * (1 + r[co1]) * asset_grid1 * (1 - remmittance_savings_rate)
                           + Profits[co2] / Czship[co2])/price[co1]
                    
                    rel_price = (price[co1]*(1-remmittance_savings_rate)+price[co2]*remmittance_savings_rate) / price[co1]
                    upper_bound_applied = np.minimum(real_income/rel_price,upper_bound)
                    fut_asset1, V_val_temp = ip.goldenx(current_util,lower_bound,upper_bound_applied,max_tol,real_income,coeff_e[
                        :,index_tod],rel_price)
                    Phi_prime_store[:,:,index_tod] = ip.spli_basex(param_asset,fut_asset1.flatten(),deg = degree,knots = knots_a)
                    #Phi_prime_store[:,:,index_tod] = ip.spli_basex(param_asset,fut_asset1.flatten(),deg = degree)
                    V_val[:,np.newaxis,index_tod] = V_val_temp
                    #coeff_next[:,index_tod] = np.linalg.solve(Phi,V_val_temp)[:,0]
        V_bar_temp = np.zeros((param_asset[0],No_countries**2 * Skill_level_no,No_countries))
        for co in range(No_countries):
            V_bar_temp[:,:,co] = V_val @ Transition_Matrix_stacked1[:,:,co].T - Util_cost_matrix[:,:,co].T
        index_max = V_bar_temp.argmax(axis =2)
        V_bar_max = ((index_max == 0 * np.ones(index_max.shape)) * (V_val @ Transition_Matrix_stacked1[:,:,0].T 
                                                                        - Util_cost_matrix[:,:,0].T))
        for co in range(1,No_countries):
            V_bar_max = V_bar_max + ((index_max == co * np.ones(index_max.shape)) * (V_val @ Transition_Matrix_stacked1[:,:,co].T 
                                                                        - Util_cost_matrix[:,:,co].T))
        V_bar = freedom_fairy *V_bar_max + (1 - freedom_fairy) * V_val @ Transition_Matrix_stacked_stay.T
        
        #coeff, conv1 = Bellman_iter(coeff,coeff_next,dampen)
        #coeff_e, conv = Bellman_iter(coeff_e,coeff_next_e,dampen,V_bar,index_max)
        #coeff_e, conv = Bellman_iter2(coeff_e,coeff_next_e,dampen,V_bar)
        #coeff_e, conv = Bellman_iter2(coeff_e,dampen,V_bar)
        #coeff_e, conv = Newton_iter2(coeff_e,dampen,V_bar,index_max,Phi,Phi_prime_store)
        if(iterate < fastcoeff):
            coeff_e, conv = Bellman_iter2(coeff_e,dampen,V_bar)
        else:
            coeff_e, conv = Newton_iter2(coeff_e,dampen,V_bar,index_max,Phi,Phi_prime_store)
        if(conv > 100000000000000):
            conv = 0
            print(conv,'Really bad guess - Bellman iteration does not converge',xsol)
        else:
            conv = ((V_bar_prev - V_bar)**2).mean()

#        elif(iterate > 2 * fastcoeff):
#            conv = 0
#            print('Really bad guess - Bellman iteration does not converge',xsol)
#        else:
#            coeff_e, conv = Newton_iter2(coeff_e,dampen,V_bar,index_max,Phi,Phi_prime_store)
        iterate = iterate +1
        
        V_bar_prev = V_bar.copy()
        print(conv)
    if (conv == 0 ):
        residual = 1e+15 * xsol
    else:
        #Aggregates    
        Q_store = np.zeros((fine_grid_no,fine_grid_no,No_countries**2 * Skill_level_no))
        V_val_fine = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no))
        cons_fine = np.zeros((fine_grid_no*No_countries**2 * Skill_level_no,1))
        asset_prime_fine = np.zeros((fine_grid_no*No_countries**2 * Skill_level_no,1))
        for co2 in range(No_countries):
            for co1 in range(No_countries):
                for skill_it in range(Skill_level_no):
                    index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
                    real_income = (Wage[co1] * Skill_level1[skill_it,co1] + 
                       price[co2] * (1 + r[co2]) * asset_grid_fine1 * remmittance_savings_rate
                       + 
                       price[co1] * (1 + r[co1]) * asset_grid_fine1 * (1 - remmittance_savings_rate)
                       + Profits[co2] / Czship[co2])/price[co1]
                    
                    rel_price = (price[co1]*(1-remmittance_savings_rate)+price[co2]*remmittance_savings_rate) / price[co1]
                    upper_bound_applied = np.minimum(real_income/rel_price,upper_bound_fine)
                    fut_asset1, V_val_temp = ip.goldenx(current_util,lower_bound_fine,upper_bound_applied,max_tol,real_income,coeff_e[
                        :,index_tod],rel_price)
                    cons_fine[index_tod*fine_grid_no : (index_tod + 1)*fine_grid_no,0,np.newaxis] = real_income - rel_price* fut_asset1
                    asset_prime_fine[index_tod*fine_grid_no : (index_tod + 1)*fine_grid_no,0,np.newaxis] = fut_asset1
                    V_val_fine[:,np.newaxis,index_tod] = V_val_temp
                    #Q_store[:,:,index_tod] = ip.spli_basex((fine_grid_no,param_asset[1],param_asset[2]),fut_asset1.flatten(),knots = knots_a_fine,deg = 1)
                    Q_store[:,:,index_tod] = ip.spli_basex((fine_grid_no,param_asset[1],param_asset[2]),fut_asset1.flatten(),deg = degree_fine,knots = asset_grid_fine)
        V_bar_temp = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no,No_countries))
        for co in range(No_countries):
            V_bar_temp[:,:,co] = V_val_fine @ Transition_Matrix_stacked1[:,:,co].T - Util_cost_matrix_fine[:,:,co].T
        
        index_max = V_bar_temp.argmax(axis =2)
#        BigT = np.zeros((No_countries**2 * Skill_level_no * fine_grid_no,No_countries**2 * Skill_level_no * fine_grid_no))
#    
#        for ii in range(No_countries**2 * Skill_level_no):
#            index_max_local = np.kron(index_max[:,ii,None],np.ones((1,No_countries**2 * Skill_level_no * fine_grid_no)))
#            D = np.zeros(index_max_local.shape)
#            for co in range(No_countries):
#                D = D  + (index_max_local == co * np.ones(index_max_local.shape))*  np.kron( Transition_Matrix_stacked1[ii,:,co],Q_store[:,:,ii])
#            BigT[(ii * fine_grid_no):((ii + 1) * fine_grid_no),:] = freedom_fairy * D + (1 - freedom_fairy) * np.kron( Transition_Matrix_stacked_stay[ii,:],Q_store[:,:,ii])
        BigT = np.zeros((No_countries * Skill_level_no * fine_grid_no,No_countries * Skill_level_no * fine_grid_no,No_countries))
        for jj in range(No_countries):
            index_tod = jj*No_countries* Skill_level_no
            index_tom = (jj+1)*No_countries* Skill_level_no
            for ii in range(No_countries * Skill_level_no):
                index_max_local = np.kron(index_max[:,ii + index_tod,None],np.ones((1,No_countries * Skill_level_no * fine_grid_no)))
                D = np.zeros(index_max_local.shape)
                for co in range(No_countries):
                    D = D  + (index_max_local == co * np.ones(index_max_local.shape))*  np.kron( Transition_Matrix_stacked1[ii + index_tod,index_tod:index_tom,co],Q_store[:,:,ii+index_tod])
                BigT[(ii * fine_grid_no):((ii + 1) * fine_grid_no),:,jj] = freedom_fairy * D + (1 - freedom_fairy) * np.kron( Transition_Matrix_stacked_stay[ii + index_tod,index_tod:index_tom],Q_store[:,:,ii + index_tod])
        #BigT1 = Perturb_Q(BigT)
        #stat_distr1 = stat_distr_eigen_sep(BigT)
        stat_distr1 = stat_distr_eigen_sparse(BigT)
        #stat_distr1 = stat_distr(BigT,iterate = 1)
        #stat_distr2 = stat_distr_eigen(BigT)
        #stat_distr1 = stat_distr2[:,None]
        L_effective_trans = loc_selector @ (stat_distr1 * skill_level_tile)
        K_trans = ((remmittance_savings_rate * cit_selector + (1 - remmittance_savings_rate) * loc_selector) 
                   @ (stat_distr1 * asset_prime_fine))
        spending = loc_selector @ (stat_distr1 * cons_fine)
        print(loc_selector @ stat_distr1,xsol)
        # Residual block
        residual[:No_countries] =  L - L_effective_trans.T  #L_effective - L_effective_trans.T
        residual[No_countries:(2 * No_countries)] =  K - K_trans.T
        residual[(2 * No_countries):] =  (spending.flatten() + delta * K - production.flatten())[1:] #+ Profits/No_countries 
        print(residual,xsol)
    return residual


#xsol=np.array([ 0.81503893,  0.5213012 ,  1.23944442,  1.43549757,  0.16159252])#Working for tau=5 remittance = 0.85 - MAIN ONE
#xsol=np.array([ 0.8142743,   0.52541808,  1.20535298,  1.31169531 , 0.16034493])#Working for tau=5000 remittance = 1 - MAIN ONE
#xsol=np.array([ 0.81465235,  0.52316513,  1.23865694,  1.42127124,  0.16161807])#Working for tau=5 remittance = 0.95 - MAIN ONE
#xsol=np.array([ 0.62439584,  0.33603029,  1.3224035 ,  2.14882377,  0.10066762])#Working for tau=5000 remittance = 0.85 TFP_Domestic=0.865 - MAIN ONE

#xsol=np.array([ 0.67450235,  0.30049569,  0.95152533,  2.33352873,  0.09863678])#Working for tau=5 remittance = 0.85 TFP_Domestic=0.865 - MAIN ONE
#xsol=np.array([ 0.67376086,  0.30201691,  0.95186162,  2.3367901 ,  0.0982807 ])#Working for tau=5 remittance = 0.95 TFP_Domestic=0.865 - MAIN ONE
xsol =np.array([ 0.6867299 ,  0.31700754,  1.48241525,  2.24330171,  0.13563558])#Working for tau=5 remittance = 0.0 TFP_Domestic=0.865 - MAIN ONE
#xsol = np. array([ 0.55739596,  0.08777387, -0.49716254,  0.98224019,  0.01925981])#Working for tau=1000
#xsol = np.array([ 0.5605257 ,  0.08625873, -0.49246129,  0.9683536 ,  0.02075161])#Working for tau=0
#xsol = np.array([ 0.60401584,  0.09823977, -0.80197559,  1.31426676, -0.00817952])
#xsol = np.array([ 0.57274441,  0.07523245, -0.23441563 , 1.27451142,  0.03739227])#Working for tau=0 remittance = 0.5
#x_root1 = optimize.root(stst_resid2,xsol,options={ 'ftol': root_tol,'disp': True} ,method = 'df-sane')
def bisection(x_start = xsol,root_tol = root_tol,maxiter = 1000,resid_dampen = 0.05):
    xsol = x_start
    resid = stst_resid2(xsol)
    tol = (np.sum(resid**2))**(1/2)
    iterate1 = 0
    while (tol >root_tol and iterate1< maxiter):
        iterate1 = iterate1 +1
        xsol = xsol + np.minimum(tol,resid_dampen)  * resid * np.abs(resid)
        resid = stst_resid2(xsol)
        tol = np.sum(resid**2)**(1/2)#/iterate  
        print(tol,iterate1)
    print(np.sum(resid**2)**(1/2))
    return xsol
# Plotting stuff
from table_creator_fcn import table_creator
import seaborn as sns

def moving_avg(y):
    y1 = y.copy()
    for ii in range(2,y.shape[0]-2):
        y1[ii] = (y[ii] + y[ii+1] + y[ii+2] +y[ii-1] +y[ii-2])/5
    return y1
def figure_distr(stat_distr1,xsol,no_graphs,txt= 'closed.tex'):
    from matplotlib2tikz import save as tikz_save
    #labels = ("Skilled employed","Skilled unemployed","Unskilled employed","Unskilled unemployed")
    labels = ("Skilled employed","Unemployed","Unskilled employed")
    fontsize = 14
    fig = plt.figure(figsize=(20,20))
    if no_graphs == 2:
       ax = plt.subplot(211) 
    elif no_graphs == 3:
        ax = plt.subplot(311)
    elif no_graphs == 4:
        ax = plt.subplot(221)
    #sns.kdeplot(np.array([asset_grid_fine[:,None],stat_distr1[:1*fine_grid_no] + stat_distr1[1*fine_grid_no:2*fine_grid_no]]), shade=True)
    #sns.distplot(stat_distr1[:1*fine_grid_no] + stat_distr1[1*fine_grid_no:2*fine_grid_no], bins=200, kde=True, rug=False)
    ax.plot(asset_grid_fine,moving_avg(stat_distr1[:1*fine_grid_no] + stat_distr1[1*fine_grid_no:2*fine_grid_no]),label=labels[0])
    ax.plot(asset_grid_fine,moving_avg(stat_distr1[2*fine_grid_no:3*fine_grid_no]+stat_distr1[5*fine_grid_no:6*fine_grid_no]),label=labels[1], dashes=[3, 5])
    ax.plot(asset_grid_fine,moving_avg((stat_distr1[3*fine_grid_no:4*fine_grid_no] + stat_distr1[4*fine_grid_no:5*fine_grid_no])),label=labels[2], dashes=[6, 2])
    ax.legend(loc="center right",fontsize=fontsize)
    ax.set_title("Home citizens in Home")
    if no_graphs == 3:
        ax = plt.subplot(312)
        relative_spot = 6*fine_grid_no
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]),label=labels[0])
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]),label=labels[1], dashes=[3, 5])
        ax.plot(asset_grid_fine,moving_avg((stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)])),label=labels[2], dashes=[6, 2])
        ax.legend(loc="center right",fontsize=fontsize)
        ax.set_title("Home citizens in Foreign")
    elif no_graphs == 4:
        ax = plt.subplot(222)
        relative_spot = 6*fine_grid_no
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]),label=labels[0])
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]),label=labels[1], dashes=[3, 5])
        ax.plot(asset_grid_fine,moving_avg((stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)])),label=labels[2], dashes=[6, 2])
        ax.legend(loc="center right",fontsize=fontsize)
        ax.set_title("Home citizens in Foreign")

    if no_graphs == 4:
        ax = plt.subplot(223)
        relative_spot = 12*fine_grid_no
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]),label=labels[0])
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]),label=labels[1], dashes=[3, 5])
        ax.plot(asset_grid_fine,moving_avg((stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)])),label=labels[2], dashes=[6, 2])    
        ax.legend(loc="center right",fontsize=fontsize)
        ax.set_title("Foreign citizens in Home")
    if no_graphs == 2:
        ax = plt.subplot(212) 
        relative_spot = 18*fine_grid_no
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]),label=labels[0])
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]),label=labels[1], dashes=[3, 5])
        ax.plot(asset_grid_fine,moving_avg((stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)])),label=labels[2], dashes=[6, 2])            
    
        ax.legend(loc="center right",fontsize=fontsize)
        ax.set_title("Foreign citizens in Foreign")
    elif no_graphs == 3:
        ax = plt.subplot(313)
        relative_spot = 18*fine_grid_no
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]),label=labels[0])
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]),label=labels[1], dashes=[3, 5])
        ax.plot(asset_grid_fine,moving_avg((stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)])),label=labels[2], dashes=[6, 2])            
        
        ax.legend(loc="center right",fontsize=fontsize)
        ax.set_title("Foreign citizens in Foreign")
    elif no_graphs == 4:
        ax = plt.subplot(224)
        relative_spot = 18*fine_grid_no
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]),label=labels[0])
        ax.plot(asset_grid_fine,moving_avg(stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]),label=labels[1], dashes=[3, 5])
        ax.plot(asset_grid_fine,moving_avg((stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)])),label=labels[2], dashes=[6, 2])            
        
        ax.legend(loc="center right",fontsize=fontsize)
        ax.set_title("Foreign citizens in Foreign")
    #fig.savefig('closed.png', dpi=100)
    
    tikz_save(
        txt,
        figureheight='\\figureheight',
        figurewidth='\\figurewidth'
        )
    return xsol

def statistics_distr(stat_distr1,filename):
    # Income and wealth inequality
    mean_income = np.zeros((No_countries**2,1))
    median_income = np.zeros((No_countries**2,1))
    median_income_distr = np.zeros((No_countries**2,1))
    mean_wealth = np.zeros((No_countries**2,1))
    median_wealth = np.zeros((No_countries**2,1))
    median_wealth_distr = np.zeros((No_countries**2,1))
    GINI_income = np.zeros((No_countries**2,1))
    GINI_wealth = np.zeros((No_countries**2,1))
    mean_income_global = 0
    mean_wealth_global = 0
    GINI_income_global = 0
    GINI_wealth_global = 0
    pop = np.zeros((No_countries**2,1))
    for ii in range(No_countries):
        for jj in range(No_countries):
            medi_inc = 0
            medi_wealth = 0
            for kk in range(Skill_level_no):
                tod = ii*No_countries*Skill_level_no+jj*Skill_level_no + kk
                tom = ii*No_countries*Skill_level_no+jj*Skill_level_no
                tom1 = ii*No_countries*Skill_level_no+(jj + 1)*Skill_level_no
                if stat_distr1[(tom * (fine_grid_no)):(tom1 * (fine_grid_no)),0].sum() > 0:
                    mean_income[ii*No_countries+jj,0] = mean_income[ii*No_countries+jj,0] + stat_distr1[(tod * (fine_grid_no)):((tod +1) * (fine_grid_no)),0].sum()/ stat_distr1[(tom * (fine_grid_no)):(tom1 * (fine_grid_no)),0].sum() * Skill_level1[kk,jj]*Wage[jj]
                    tmp = median_income_distr[ii*No_countries+jj,0]
                    median_income_distr[ii*No_countries+jj,0] = tmp + stat_distr1[(tod * (fine_grid_no)):((tod +1) * (fine_grid_no)),0].sum() / stat_distr1[(tom * (fine_grid_no)):(tom1 * (fine_grid_no)),0].sum()
                    if median_income_distr[ii*No_countries+jj,0] > 0.5 and medi_inc == 0:
                        #median_income[ii*No_countries+jj,0] = ((median_income_distr[ii*No_countries+jj,0] - 0.5)/(median_income_distr[ii*No_countries+jj,0] - tmp) * Skill_level1[kk,jj]*Wage[jj] +
                        #             ((0.5 - tmp)/(median_income_distr[ii*No_countries+jj,0] - tmp) * Skill_level1[kk-1,jj]*Wage[jj]))
                        median_income[ii*No_countries+jj,0] = Skill_level1[kk,jj]*Wage[jj]
                        medi_inc = 1
                    mean_wealth[ii*No_countries+jj,0] = mean_wealth[ii*No_countries+jj,0] + asset_grid_fine @ stat_distr1[(tod * (fine_grid_no)):((tod +1) * (fine_grid_no)),0]/stat_distr1[(tom * (fine_grid_no)):(tom1 * (fine_grid_no)),0].sum() * Skill_level1[kk,jj]*Wage[jj]
                
            for ll in range(fine_grid_no):
                tod = (ii*No_countries*Skill_level_no+jj*Skill_level_no + np.arange(0,Skill_level_no))
                tmp1 = median_wealth_distr[ii*No_countries+jj,0]
                if(stat_distr1[(tod[0]*fine_grid_no):((tod[-1] + 1)*fine_grid_no),0].sum() > 0):
                    median_wealth_distr[ii*No_countries+jj,0] = tmp1 + stat_distr1[tod*fine_grid_no  + ll,0].sum() / stat_distr1[(tod[0]*fine_grid_no):((tod[-1] + 1)*fine_grid_no),0].sum()
                    if median_wealth_distr[ii*No_countries+jj,0] > 0.5 and medi_wealth == 0:
                        #print(median_wealth_distr[ii*No_countries+jj,0])
                        median_wealth[ii*No_countries+jj,0] = asset_grid_fine[ll]
                        medi_wealth = 1
    for ii in range(No_countries):
        for jj in range(No_countries):
            for kk in range(Skill_level_no):
                tod = ii*No_countries*Skill_level_no+jj*Skill_level_no + kk
                tom = ii*No_countries*Skill_level_no+jj*Skill_level_no
                tom1 = ii*No_countries*Skill_level_no+(jj + 1)*Skill_level_no
                sum_stat_distr = stat_distr1[(tom * (fine_grid_no)):(tom1 * (fine_grid_no)),0].sum()
                if sum_stat_distr > 0:
                    for kkk in range(Skill_level_no):
                        tod2 = ii*No_countries*Skill_level_no+jj*Skill_level_no + kkk
                        GINI_income[ii*No_countries+jj,0] = GINI_income[ii*No_countries+jj,0] + (stat_distr1[(tod * (fine_grid_no)):((tod +1) * (fine_grid_no)),0].sum()/ sum_stat_distr *
                                   stat_distr1[(tod2 * (fine_grid_no)):((tod2 +1) * (fine_grid_no)),0].sum()/ sum_stat_distr *
                                   np.absolute(Skill_level1[kk,jj]*Wage[jj] - Skill_level1[kkk,jj]*Wage[jj]))
                        GINI_income_global = GINI_income_global + (stat_distr1[(tod * (fine_grid_no)):((tod +1) * (fine_grid_no)),0].sum() *
                                   stat_distr1[(tod2 * (fine_grid_no)):((tod2 +1) * (fine_grid_no)),0].sum() *
                                   np.absolute(Skill_level1[kk,jj]*Wage[jj] - Skill_level1[kkk,jj]*Wage[jj])
                        )
    
            for ll in range(fine_grid_no):
                tod = (ii*No_countries*Skill_level_no+jj*Skill_level_no + np.arange(0,Skill_level_no))
                sum_stat_distr = stat_distr1[(tod[0]*fine_grid_no):((tod[-1] + 1)*fine_grid_no),0].sum()
                if(sum_stat_distr > 0):
                    for lll in range(fine_grid_no):
                        GINI_wealth[ii*No_countries+jj,0] = GINI_wealth[ii*No_countries+jj,0] + ( np.abs(asset_grid_fine[ll] - asset_grid_fine[lll]) * 
                                   stat_distr1[tod*fine_grid_no  + ll,0].sum() / sum_stat_distr * stat_distr1[tod*fine_grid_no  + lll,0].sum() / sum_stat_distr)
                        GINI_wealth_global = GINI_wealth_global + ( np.abs(asset_grid_fine[ll] - asset_grid_fine[lll]) * 
                                   stat_distr1[tod*fine_grid_no  + ll,0].sum()  * stat_distr1[tod*fine_grid_no  + lll,0].sum() )
    GINI_income = GINI_income/2/mean_income
    GINI_wealth = GINI_wealth/2/mean_wealth
    for ii in range(No_countries):
        for jj in range(No_countries): 
            tom = ii*No_countries*Skill_level_no+jj*Skill_level_no
            tom1 = ii*No_countries*Skill_level_no+(jj + 1)*Skill_level_no
            pop[ii*No_countries + jj] = pop[ii*No_countries + jj] + stat_distr1[(tom * (fine_grid_no)):(tom1 * (fine_grid_no)),0].sum()
            #print(stat_distr1[(tom * (fine_grid_no)):(tom1 * (fine_grid_no)),0].sum())
    mean_income_global = pop.T @ mean_income   
    mean_wealth_global = pop.T @ mean_wealth     
    GINI_income_global = GINI_income_global/2/mean_income_global
    GINI_wealth_global = GINI_wealth_global/2/mean_wealth_global
    np.savetxt(filename + '_stat_distr.csv', stat_distr1, delimiter=",")
    file = open(filename + '.txt','w') 
    file.write('Aggregates: \n') 
    file.write('Population distribution: \n' + str(100*loc_selector @ stat_distr1) + '\n')
    file.write('Per capita output: \n'+ str(production.flatten() / (loc_selector @ stat_distr1).flatten())+ '\n')
    file.write('K/Y ration: \n'+ str(K / production)+ '\n')
    file.write('Home citizens living abroad in % \n'+ str(100*stat_distr1[6*fine_grid_no:12*fine_grid_no].sum() / Czship[0])+ '\n')
    file.write('Foreign citizens living abroad in % \n'+ str(100*stat_distr1[12*fine_grid_no:18*fine_grid_no].sum() / Czship[1])+ '\n')
    file.write('Ownership of capital by emigrants: \n'+ str(100*remmittance_savings_rate * (np.tile(asset_grid_fine1,(6,1)).T @ (stat_distr1[6*fine_grid_no:12*fine_grid_no]))/K[0] ) + str(100*remmittance_savings_rate * (np.tile(asset_grid_fine1,(6,1)).T @ (stat_distr1[12*fine_grid_no:18*fine_grid_no]))/K[1] )+'\n')
    file.write('Wages: \n'+ str(Wage)+ '\n')
    file.write('Price: \n'+ str(price)+ '\n')
    file.write('Return on capital: \n'+ str(r)+ '\n')
    file.write('Inequality measures: \n') 
    file.write('Mean income: \n'+ str(mean_income)+ '\n')
    file.write('Median income: \n'+ str(median_income)+ '\n')
    file.write('Mean wealth: \n'+ str(mean_wealth)+ '\n')
    file.write('Median wealth: \n'+ str(median_wealth)+ '\n')
    file.write('GINI income: \n'+ str(GINI_income)+ '\n')
    file.write('GINI wealth: \n'+ str(GINI_wealth)+ '\n')
    file.write('GINI income global: \n'+ str(GINI_income_global)+ '\n')
    file.write('GINI wealth global: \n'+ str(GINI_wealth_global)+ '\n')

     
    
    file.close() 
    return xsol
#figure_distr(stat_distr1,xsol,2,txt= 'closed.tex')
#statistics_distr(stat_distr1,'closed')    
#figure_distr(stat_distr1,xsol,4,txt= 'open_full_remittance.tex')
#statistics_distr(stat_distr1,'open_full_remittance')  
#figure_distr(stat_distr1,xsol,3,txt= 'open_085_remittance.tex')
#statistics_distr(stat_distr1,'open_085_remittance')  
#figure_distr(stat_distr1,xsol,4,txt= 'open_half_remittance.tex')
#statistics_distr(stat_distr1,'open_half_remittance')  
  #figure_distr(stat_distr1,xsol,3,txt= 'open_095_remittance.tex') 
#statistics_distr(stat_distr1,'open_095_remittance')  
#figure_distr(stat_distr1,xsol,3,txt= 'open_0_remittance.tex') 
#statistics_distr(stat_distr1,'open_0_remittance')  
 #statistics_distr(stat_distr1,'open_085_remittanceTEST') 
#statistics_distr(stat_distr1,'closedTEST')   
compile_all_the_data = 0
# Loading all the different scenarios
if compile_all_the_data == 1:
    # Create histograms:
    N = 2 # Cases to compare
    casenames = ('No migration', 'Migration')
    #casenames = ('Baseline', 'Higher remittances')
    skill_labels = ('Skilled', 'Unemployed', 'Unskilled')
    #figure_skill_name = 'skill_distribution_ext1.tikz'
    figure_skill_name = 'skill_distribution_baseline.tikz'
    # Create statistics tables:
    filename = 'statistics_baseline'
    #filename = 'statistics_ext1'
    use_statistics_name0 = 'closed.txt'
    use_statistics_name1 = 'open_085_remittance.txt'
#    use_statistics_name0 = 'open_085_remittance.txt'
#    use_statistics_name1 = 'open_095_remittance.txt'
    
    fontsize = 14
    skill_dist_home = np.zeros((3,N))
    skill_dist_home_migrants = np.zeros((3,N))
    skill_dist_foreign = np.zeros((3,N))
    skill_dist_foreign_migrants = np.zeros((3,N))
    
    open_full_remittance_stat_distr1 = np.genfromtxt('open_085_remittance_stat_distr.csv',delimiter=',')   #np.genfromtxt('open_095_remittance_stat_distr.csv',delimiter=',')     #
    stat_distr1 = open_full_remittance_stat_distr1.copy()
    
    relative_spot = 0*fine_grid_no
    relative_spot_skill = 1
    skill_dist_home[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
    skill_dist_home[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
    skill_dist_home[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()
   
    relative_spot = 6*fine_grid_no
    skill_dist_home_migrants[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
    skill_dist_home_migrants[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
    skill_dist_home_migrants[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()
    
    relative_spot = 18*fine_grid_no
    skill_dist_foreign[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
    skill_dist_foreign[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
    skill_dist_foreign[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()

    relative_spot = 12*fine_grid_no
    skill_dist_foreign_migrants[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
    skill_dist_foreign_migrants[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
    skill_dist_foreign_migrants[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()
  
#    open_full_remittance_stat_distr1 = np.genfromtxt('open_085_remittance_stat_distr.csv',delimiter=',')     #= np.genfromtxt('closed_stat_distr.csv',delimiter=',')   
#    stat_distr1 = open_full_remittance_stat_distr1.copy()
#    
#    relative_spot = 0*fine_grid_no
#    relative_spot_skill = 0
#    skill_dist_home[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
#    skill_dist_home[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
#    skill_dist_home[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()
#   
#    relative_spot = 6*fine_grid_no
#    skill_dist_home_migrants[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
#    skill_dist_home_migrants[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
#    skill_dist_home_migrants[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()
#    
#    relative_spot = 18*fine_grid_no
#    skill_dist_foreign[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
#    skill_dist_foreign[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
#    skill_dist_foreign[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()
#
#    relative_spot = 12*fine_grid_no
#    skill_dist_foreign_migrants[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
#    skill_dist_foreign_migrants[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
#    skill_dist_foreign_migrants[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()
          
    #closed_stat_distr1 = np.genfromtxt('open_085_remittance_stat_distr.csv',delimiter=',')     
    closed_stat_distr1 = np.genfromtxt('closed_stat_distr.csv',delimiter=',') 
    stat_distr1 = closed_stat_distr1.copy()
    relative_spot_skill = 0
    relative_spot = 0*fine_grid_no
    skill_dist_home[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
    skill_dist_home[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
    skill_dist_home[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()
       
    relative_spot = 18*fine_grid_no
    skill_dist_foreign[0,relative_spot_skill] = (stat_distr1[relative_spot:(relative_spot + 1*fine_grid_no)] + stat_distr1[(relative_spot + 1*fine_grid_no):(relative_spot + 2*fine_grid_no)]).sum()
    skill_dist_foreign[1,relative_spot_skill] = (stat_distr1[(relative_spot + 2*fine_grid_no):(relative_spot + 3*fine_grid_no)]+stat_distr1[(relative_spot + 5*fine_grid_no):(relative_spot + 6*fine_grid_no)]).sum()
    skill_dist_foreign[2,relative_spot_skill] = (stat_distr1[(relative_spot + 3*fine_grid_no):(relative_spot + 4*fine_grid_no)] + stat_distr1[(relative_spot + 4*fine_grid_no):(relative_spot + 5*fine_grid_no)]).sum()

    ind = np.arange(N)  # the x locations for the groups
    width = 0.25       # the width of the bars

    fig = plt.figure(figsize=(20,10))
    #ax = fig.add_subplot(221)
    ax = fig.add_subplot(311)
    rects1 = ax.bar(ind,skill_dist_home[0,:], width, color='royalblue')
    rects2 = ax.bar(ind+width, skill_dist_home[1,:], width, color='seagreen')
    rects3 = ax.bar(ind + 2 * width, skill_dist_home[2,:], width, color='red')
    # add some labels
    ax.set_ylabel('Measure',fontsize=fontsize)
    #ax.set_xlabel('Time',fontsize=fontsize)
    ax.set_title('Distribution of skill in Home for locals',fontsize=fontsize)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels( casenames ,fontsize=fontsize)
    
    ax.legend( (rects1[0], rects2[0],rects3[0]), skill_labels,loc="best",fontsize=fontsize )
    #fig.savefig('Home_distr.png', dpi=100)
    #Foreign
    #ax1 = fig.add_subplot(222)
    ax1 = fig.add_subplot(312)
    rects1 = ax1.bar(ind,skill_dist_foreign[0,:], width, color='royalblue')
    rects2 = ax1.bar(ind+width, skill_dist_foreign[1,:], width, color='seagreen')
    rects3 = ax1.bar(ind + 2 * width, skill_dist_foreign[2,:], width, color='red')
    # add some labels
    ax1.set_ylabel('Measure',fontsize=fontsize)
    #ax1.set_xlabel('Time',fontsize=fontsize)
    ax1.set_title('Distribution of skill in Foreign for locals',fontsize=fontsize)
    ax1.set_xticks(ind + width / 2)
    ax1.set_xticklabels( casenames ,fontsize=fontsize)
    
    ax1.legend( (rects1[0], rects2[0],rects3[0]), skill_labels,loc="best",fontsize=fontsize )
    #Home citizens in Foreign
    #ax2 = fig.add_subplot(223)
    ax2 = fig.add_subplot(313)
    rects1 = ax2.bar(ind,skill_dist_home_migrants[0,:], width, color='royalblue')
    rects2 = ax2.bar(ind+width, skill_dist_home_migrants[1,:], width, color='seagreen')
    rects3 = ax2.bar(ind + 2 * width, skill_dist_home_migrants[2,:], width, color='red')
    # add some labels
    ax2.set_ylabel('Measure',fontsize=fontsize)
    #ax2.set_xlabel('Time',fontsize=fontsize)
    ax2.set_title('Distribution of skill in Foreign for migrants',fontsize=fontsize)
    ax2.set_xticks(ind + width / 2)
    ax2.set_xticklabels( casenames ,fontsize=fontsize)
    
    ax2.legend( (rects1[0], rects2[0],rects3[0]), skill_labels,loc="best",fontsize=fontsize )
    
        #Foreign citizens in Home
    #ax3 = fig.add_subplot(224)
#    rects1 = ax3.bar(ind,skill_dist_foreign_migrants[0,:], width, color='royalblue')
#    rects2 = ax3.bar(ind+width, skill_dist_foreign_migrants[1,:], width, color='seagreen')
#    rects3 = ax3.bar(ind + 2 * width, skill_dist_foreign_migrants[2,:], width, color='red')
#    # add some labels
#    ax3.set_ylabel('Measure',fontsize=fontsize)
#    #ax3.set_xlabel('Time',fontsize=fontsize)
#    ax3.set_title('Distribution of skill in Home for migrants',fontsize=fontsize)
#    ax3.set_xticks(ind + width / 2)
#    ax3.set_xticklabels( casenames ,fontsize=fontsize)
#    
#    ax3.legend( (rects1[0], rects2[0],rects3[0]), skill_labels,loc="best",fontsize=fontsize )
    fig.savefig(figure_skill_name[:-5] +'.png', dpi=100)
#    from matplotlib2tikz import save as tikz_save
#    tikz_save(
#    figure_skill_name,
#    figureheight='\\figureheight',
#    figurewidth='\\figurewidth')
    

    success = table_creator(filename,use_statistics_name0,use_statistics_name1,migration_baseline = 0)