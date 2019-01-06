import pandas as pd
import numpy as np
import scipy as sp
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import requests, zipfile, io # So that we can download and unzip files\n",
#from matplotlib2tikz import save as tikz_save
import matplotlib
import statsmodels.tsa.filters as filters
import quandl as Quandl
import quantecon as qe
#import warnings; warnings.simplefilter('ignore')
import multiprocessing    
import Multiprocessfile_gridsearch
import Multiprocessfile_aggregates_gridsearch
import Multiprocessfile_Newton_gridsearch
from functools import partial
from poly_base import interpolate as ip
from income_process import income_process
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

citizenship_attrition = 0.000 # half a percent chance of becoming local citizen
No_countries = 2
freedom_fairy = 0.5
#freedom_fairy1=[0.5,1]
remmittance_savings_rate = 0.2 #proportion of your assets held in the location of your citizenship
TFP_Domestic = 0.865
TFP_Foreign = 1.0
fastcoeff = 2 # Number of Bellman iterations
alpha = np.array([0.43,0.33])#([0.432,0.33])
alpha_l = np.array([0.5,0.5])#([0.498,0.6]) # elasticity on labor in the production function
Total_Labor_Domestic = 0.35
Total_Labor_Foreign = 0.65
degree = 1 # spline approximation - cubic
TFP = np.array([TFP_Domestic] + [TFP_Foreign])
Czship = np.array([Total_Labor_Domestic] + [Total_Labor_Foreign])
AdjustCost = np.zeros((No_countries**2))
AdjustCost[0] = 1
AdjustCost[1] = 1.
AdjustCost[2] = 1.
AdjustCost[3] = 1
curv_liquid = 1.0
curv_illiquid = 0.15#1.0
AdjustCost_quad = 0.00*AdjustCost
AdjustCost = 0.00 * AdjustCost
AdjustCost_curve = 0.0
CapLimiter = 0.025

Skill_level1,Skill_Matrix_stacked,Skill_level_no,Transition_Matrix = income_process()  #,prod_norm_level
Skill_Matrix_stacked1 = np.tile(Skill_Matrix_stacked,(No_countries,1))
Skill_level_stacked = Skill_level1.T.flatten().reshape((No_countries*Skill_level_no,1))
liquid_param_asset = (3,-0.1,0.1)

#asset_grid = ip.wealth_knot(param_asset)
#liquid_asset_grid = np.linspace(liquid_param_asset[1],liquid_param_asset[2],liquid_param_asset[0])
liquid_asset_grid = ip.wealth_knot(liquid_param_asset, degree=degree, curv=curv_liquid,lower_bound = -liquid_param_asset[1] + 1e-7)
knots_liquid = ip.wealth_knot(liquid_param_asset, degree=degree, curv=curv_liquid,lower_bound = -liquid_param_asset[1] + 1e-7)
liquid_fine_grid_no = 5
liquid_param_asset_fine =(liquid_fine_grid_no,liquid_param_asset[1],liquid_param_asset[2])


liquid_asset_grid_fine = ip.equidistant_nonlin_grid(liquid_asset_grid,liquid_fine_grid_no) # np.linspace(liquid_param_asset[1],liquid_param_asset[2],liquid_param_asset_fine[0])
#asset_grid_fine =ip.wealth_knot((fine_grid_no,param_asset[1],param_asset[2]))
liquid_asset_grid1 = liquid_asset_grid.reshape((liquid_param_asset[0],1))
liquid_asset_grid_fine1 = liquid_asset_grid_fine.reshape((liquid_param_asset_fine[0],1))


illiquid_param_asset = (200,0,40)
#asset_grid = ip.wealth_knot(param_asset)
illiquid_asset_grid = ip.wealth_knot(illiquid_param_asset, degree=degree, curv=curv_illiquid)
#illiquid_asset_grid = np.linspace(illiquid_param_asset[1],illiquid_param_asset[2],illiquid_param_asset[0])
illiquid_fine_grid_no = 200
illiquid_param_asset_fine = (illiquid_fine_grid_no,illiquid_param_asset[1],illiquid_param_asset[2])
illiquid_asset_grid_fine = ip.equidistant_nonlin_grid(illiquid_asset_grid,illiquid_fine_grid_no) # np.linspace(illiquid_param_asset[1],illiquid_param_asset[2],illiquid_param_asset_fine[0])
#asset_grid_fine =ip.wealth_knot((fine_grid_no,param_asset[1],param_asset[2]))
illiquid_asset_grid1 = illiquid_asset_grid.reshape((illiquid_param_asset[0],1))
illiquid_asset_grid_fine1 = illiquid_asset_grid_fine.reshape((illiquid_param_asset_fine[0],1))
knots_illiquid = ip.wealth_knot(illiquid_param_asset, degree=degree, curv=curv_illiquid)
s = np.concatenate((np.kron(np.ones((liquid_param_asset[0],1)),illiquid_asset_grid1),np.kron(liquid_asset_grid1,np.ones((illiquid_param_asset[0],1)))),1)
s_fine = np.concatenate((np.kron(np.ones((liquid_fine_grid_no,1)),illiquid_asset_grid_fine1),np.kron(liquid_asset_grid_fine1,np.ones((illiquid_fine_grid_no,1)))),1)
AssetMultiplied = liquid_param_asset[0]*illiquid_param_asset[0]
fine_grid_no = illiquid_fine_grid_no * liquid_fine_grid_no
loc_selector = np.kron(np.ones((1,No_countries)),np.kron(np.eye(No_countries,No_countries),np.ones((1,Skill_level_no*fine_grid_no)))) 
# citizenship selector matrix:
cit_selector = np.kron(np.kron(np.eye(No_countries,No_countries),np.ones((1,Skill_level_no*fine_grid_no))),np.ones((1,No_countries)))

illiquid_asset_grid_fine1_tile = np.tile(illiquid_asset_grid_fine1,(liquid_fine_grid_no * Skill_level_no * No_countries**2,1))
liquid_asset_grid_fine1_tile =np.repeat(liquid_asset_grid_fine1,illiquid_fine_grid_no).reshape((fine_grid_no,1))# np.tile(liquid_asset_grid_fine1,(illiquid_fine_grid_no * Skill_level_no * No_countries**2,1))
liquid_asset_grid_fine1_tile = np.tile(liquid_asset_grid_fine1_tile,(Skill_level_no * No_countries**2,1))
#skill_level_tile = np.tile(np.repeat(Skill_level,fine_grid_no),(1,No_countries**2)).T
skill_level_tile = np.tile(np.repeat(Skill_level_stacked,fine_grid_no),(1,No_countries)).T

init_distr = (np.kron(np.eye(No_countries).reshape((No_countries**2,1)),np.ones((  Skill_level_no * fine_grid_no,1))) / (No_countries
        * Skill_level_no * fine_grid_no)) # Initially all citizens live in their location. Only has an effect if there is absorbing state
Transition_Matrix1 = np.kron(np.eye(No_countries),Transition_Matrix[:Skill_level_no,:Skill_level_no]) + np.kron(
    (np.ones((No_countries,No_countries)) - np.eye(No_countries)),Transition_Matrix[Skill_level_no:,:Skill_level_no] )

Transition_Matrix_stacked = np.zeros((No_countries * Skill_level_no,No_countries*Skill_level_no))
Transition_Matrix_stacked1 = np.zeros((No_countries**2 * Skill_level_no,No_countries**2 *Skill_level_no,No_countries))
Transition_Matrix_stacked_stay = np.zeros((No_countries**2 * Skill_level_no,No_countries**2 *Skill_level_no))
Util_cost_matrix = tau * np.ones((No_countries**2 * Skill_level_no,AssetMultiplied,No_countries))
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

        
Phi_illiquid = ip.spli_basex(illiquid_param_asset,s[:,0],deg = degree,knots = knots_illiquid)
Phi_liquid = ip.spli_basex(liquid_param_asset,s[:,1],deg = degree,knots = knots_liquid)
Phi_fine_illiquid = ip.spli_basex(illiquid_param_asset,s_fine[:,0],deg = degree,knots = knots_illiquid )
Phi_fine_liquid = ip.spli_basex(liquid_param_asset,s_fine[:,1],deg = degree,knots = knots_liquid)

#Phi = ip.spli_basex(param_asset,asset_grid,knots = knots_a)
Phi = ip.dprod(Phi_liquid,Phi_illiquid)
Phi_fine = ip.dprod(Phi_fine_liquid,Phi_fine_illiquid)
#Phi_fine = ip.spli_basex(param_asset,asset_grid_fine,knots = knots_a_fine)
Phi_inv = np.linalg.inv(Phi)
Phi_inv_tile = np.tile(Phi_inv, (1,No_countries**2 * Skill_level_no))
#diag_transf_v = np.kron(np.ones((No_countries**2 * Skill_level_no,1)),np.eye(param_asset[0]))
def diag_mat_block(Mat):
    res = np.zeros((Mat.shape[0] * Mat.shape[1],Mat.shape[1]))
    for ii in range(Mat.shape[1]):
        res[(ii*Mat.shape[0]):((ii + 1)*Mat.shape[0]),ii] = Mat[:,ii]
    return res

illiquid_lower_bound = illiquid_param_asset[1] * np.ones((AssetMultiplied))
illiquid_lower_bound_fine = illiquid_param_asset[1] * np.ones((fine_grid_no))
illiquid_upper_bound = illiquid_param_asset[2] * np.ones((AssetMultiplied))
illiquid_upper_bound_fine = illiquid_param_asset[2] * np.ones((fine_grid_no))


liquid_lower_bound = liquid_param_asset[1] * np.ones((AssetMultiplied))
liquid_lower_bound_fine = liquid_param_asset[1] * np.ones((fine_grid_no))
liquid_upper_bound = liquid_param_asset[2] * np.ones((AssetMultiplied))
liquid_upper_bound_fine = liquid_param_asset[2] * np.ones((fine_grid_no))

def Perturb_Q(A):
    n_ds = A.shape[0]
    #Perturb Q
    eta =( A[A>0].min() )/(2.*n_ds)
    A = np.where(A<=0.0, A + eta , A)
    #Normalize Q
    for i in range(n_ds):
        A[i,:] = A[i,:]/(np.sum(A[i,:]))
    return A
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
    #print(cons.min())
    res = cons > 0
    res1 = np.absolute(cons)
    return res1**(1 - gamma) / (1 - gamma) * res - (1-res) * (10000000 * res1) 
def Bellman_iter2(coeff,dampen,V,Phi_inv = Phi_inv_tile):
    coeff_next = Phi_inv @ diag_mat_block(V)
    conv = np.max(np.absolute(coeff_next - coeff))
    #print(np.mean(np.absolute(coeff_next - coeff),1))
    #print(np.unravel_index(np.argmax(np.absolute(coeff_next - coeff), axis=None), coeff.shape))
    return dampen * coeff_next + (1 - dampen) * coeff ,conv

def Newton_iter2(coeff,dampen,V_bar,index_max,Phi,Phi_prime_store_m,Phi_prime_store_a):
    coeff_next = coeff.copy()
    g = np.zeros((coeff.shape[0] * coeff.shape[1],1))
    D = np.zeros((g.shape[0],g.shape[0]))
    g[:,0] = np.kron(np.eye(No_countries**2 * Skill_level_no),Phi) @ coeff.flatten(order='F') - V_bar.flatten(order='F')
    loc_kron = np.kron(index_max.flatten(order='F').reshape((No_countries**2 * Skill_level_no * AssetMultiplied,1)),np.ones((1,No_countries**2 * Skill_level_no * AssetMultiplied)))
    Phi_prime_kron = D.copy()
    #Phi_prime_diag = D.copy()
    for ii in range(No_countries**2 * Skill_level_no):
        MatPhi_prime = ip.dprod(Phi_prime_store_m[:,:,ii],Phi_prime_store_a[:,:,ii])
        #MatPhi_prime = Phi_prime_store[:,:,ii]
        Phi_prime_kron[:,(ii * AssetMultiplied):((ii + 1) * AssetMultiplied) ] = np.tile(MatPhi_prime
                                                                                       ,(No_countries**2 * Skill_level_no,1))
        #Phi_prime_diag[(ii * AssetMultiplied):((ii + 1) * AssetMultiplied),(ii * AssetMultiplied):((ii + 1) * AssetMultiplied)
         #             ] = MatPhi_prime
        
    for co in range(No_countries):
        D = D  + (loc_kron == co * np.ones(loc_kron.shape)) * np.kron(
            Transition_Matrix_stacked1[:,:,co],np.ones((AssetMultiplied,AssetMultiplied))) *Beta * Phi_prime_kron
    D = freedom_fairy * D + (1 - freedom_fairy) *np.kron(
    Transition_Matrix_stacked_stay[:,:],np.ones((AssetMultiplied,AssetMultiplied))) * Beta *Phi_prime_kron
    D = np.kron(np.eye(No_countries**2 * Skill_level_no),Phi) - D
    improvement = np.linalg.solve(D, g)
    for co2 in range(No_countries):
        for co1 in range(No_countries):
            for skill_it in range(Skill_level_no):
                index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
                coeff_next[:,index_tod] = coeff[:,index_tod] - improvement[(index_tod *AssetMultiplied) : ((index_tod + 1) *AssetMultiplied),0]
    conv = np.max(np.absolute(coeff_next - coeff))
    #print(np.unravel_index(np.argmax(np.absolute(coeff_next - coeff), axis=None), coeff.shape))
    return dampen * coeff_next + (1 - dampen) * coeff ,conv

#Set up the simple model for a good initial guess using the parametrization in the model
def util_simple(cons):
    return cons**(1 - gamma) / (1 - gamma)
Phi_illiquid_simple = Phi_illiquid[:illiquid_param_asset[0],:]
Phi_illiquid_inv = np.linalg.inv(Phi_illiquid_simple)
Phi_illiquid_inv.shape
Phi_inv_tile = np.tile(Phi_illiquid_inv, (1,No_countries**2 * Skill_level_no))
def current_util_simple(fut_asset,income,coeff1,rel_price):
    #fut_asset = rel_price * fut_asset
    #Phi_asset = ip.spli_basex(param_asset,fut_asset.flatten(),knots = knots_a)
    Phi_asset = ip.spli_basex(illiquid_param_asset,fut_asset.flatten(),deg = degree,knots = knots_illiquid)
    V_fut = (Phi_asset @ coeff1).reshape(fut_asset.shape)
    return util(income - rel_price * fut_asset) + Beta * V_fut

def Bellman_iter(coeff,dampen,V_bar):
    coeff_next = Phi_inv_tile @ diag_mat_block(V_bar)
    conv = np.max(np.absolute(coeff_next - coeff))
    #print(np.mean(np.absolute(coeff_next - coeff),1))
    return dampen * coeff_next + (1 - dampen) * coeff ,conv

def Newton_iter(coeff,dampen,V_bar,index_max,Phi,Phi_prime_store):
    coeff_next = coeff.copy()
    g = np.zeros((coeff.shape[0] * coeff.shape[1],1))
    D = np.zeros((g.shape[0],g.shape[0]))
    g[:,0] = np.kron(np.eye(No_countries**2 * Skill_level_no),Phi_illiquid_simple) @ coeff.flatten(order='F') - V_bar.flatten(order='F')
    loc_kron = np.kron(index_max.flatten(order='F').reshape((No_countries**2 * Skill_level_no * illiquid_param_asset[0],1)),np.ones((1,No_countries**2 * Skill_level_no * illiquid_param_asset[0])))
    Phi_prime_kron = D.copy()
    Phi_prime_diag = D.copy()
    for ii in range(No_countries**2 * Skill_level_no):
        MatPhi_prime = Phi_prime_store[:,:,ii]
        Phi_prime_kron[:,(ii * illiquid_param_asset[0]):((ii + 1) * illiquid_param_asset[0]) ] = np.tile(MatPhi_prime
                                                                                       ,(No_countries**2 * Skill_level_no,1))
        Phi_prime_diag[(ii * illiquid_param_asset[0]):((ii + 1) * illiquid_param_asset[0]),(ii * illiquid_param_asset[0]):((ii + 1) * illiquid_param_asset[0])
                      ] = MatPhi_prime
        
    for co in range(No_countries):
        D = D  + (loc_kron == co * np.ones(loc_kron.shape)) * np.kron(
            Transition_Matrix_stacked1[:,:,co],np.ones((illiquid_param_asset[0],illiquid_param_asset[0]))) *Beta * Phi_prime_kron
    D = freedom_fairy * D + (1 - freedom_fairy) *np.kron(
    Transition_Matrix_stacked_stay[:,:],np.ones((illiquid_param_asset[0],illiquid_param_asset[0]))) * Beta *Phi_prime_kron
    D = np.kron(np.eye(No_countries**2 * Skill_level_no),Phi_illiquid_simple) - D
    #improvement =  np.linalg.inv(D) @ g
    improvement = np.linalg.solve(D, g)
    for co2 in range(No_countries):
        for co1 in range(No_countries):
            for skill_it in range(Skill_level_no):
                index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
                coeff_next[:,index_tod] = coeff[:,index_tod] - improvement[(index_tod *illiquid_param_asset[0]) : ((index_tod + 1) *illiquid_param_asset[0]),0]
    conv = np.max(np.absolute(coeff_next - coeff))
    return dampen * coeff_next + (1 - dampen) * coeff ,conv
#Gridsearch for the value fcn
a_prime_gridchoice = np.tile(s_fine[:,0,None].T,(AssetMultiplied,1))
cap_a_gridchoice = np.tile(s[:,0,None],(1,fine_grid_no))

m_prime_gridchoice = np.tile(s_fine[:,1,None].T,(AssetMultiplied,1))
m_gridchoice = np.tile(s[:,1,None],(1,fine_grid_no))
income_cost = np.float_power( np.maximum((cap_a_gridchoice - a_prime_gridchoice),0),AdjustCost_curve)
cap_a_gridchoice_lim = np.tile(illiquid_upper_bound[:,None],(1,fine_grid_no))
cap_m_gridchoice_lim = np.tile(liquid_upper_bound[:,None],(1,fine_grid_no))
#Gridsearch for the stationary distribution
a_prime_gridchoice_fine = np.tile(s_fine[:,0,None].T,(fine_grid_no,1))
cap_a_gridchoice_fine = np.tile(s_fine[:,0,None],(1,fine_grid_no))

m_prime_gridchoice_fine = np.tile(s_fine[:,1,None].T,(fine_grid_no,1))
m_gridchoice_fine = np.tile(s_fine[:,1,None],(1,fine_grid_no))
income_cost_fine = np.float_power( np.maximum((cap_a_gridchoice_fine - a_prime_gridchoice_fine),0),AdjustCost_curve)
cap_a_gridchoice_lim_fine = np.tile(illiquid_upper_bound_fine[:,None],(1,fine_grid_no))
cap_m_gridchoice_lim_fine = np.tile(liquid_upper_bound_fine[:,None],(1,fine_grid_no))
def stst_resid_faster(xsol):
    log_Bond = xsol[-1]
    Bond =1/ ( np.exp(log_Bond)/ (1 + np.exp(log_Bond))* (1/Beta - 1) + 0.96)
    #Bond = 10000
    print(Bond)
    log_Wage = xsol[:No_countries]
    Wage = np.exp(log_Wage)
    log_r = xsol[No_countries:(2 * No_countries)]
    r = np.exp(log_r)/ (1 + np.exp(log_r)) * (1/Beta - 1)
    
    residual = np.zeros(xsol.shape)
    price = np.ones((No_countries))
    log_price = xsol[(2 * No_countries):-1]
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


    #Worker's problem:
    # Gridsearch:

    Adjust_mat = np.zeros((AssetMultiplied,fine_grid_no,No_countries))
    for ii in range(No_countries):
        Adjust_mat[:,:,ii] = cap_a_gridchoice * (1 + (r[ii])) > a_prime_gridchoice
    Cons_NOadj =-10000 * np.ones((AssetMultiplied,fine_grid_no,No_countries**2 * Skill_level_no))
    Cons_adj = -10000 * np.ones((AssetMultiplied,fine_grid_no,No_countries**2 * Skill_level_no)) 
    for co2 in range(No_countries):
        for co1 in range(No_countries):
            AdjustCost_applied = price[co2]*AdjustCost[co1 + co2 * No_countries]
            AdjustCost_applied_quad = price[co2]/(price[co1])*AdjustCost_quad[co1 + co2 * No_countries]
            for skill_it in range(Skill_level_no):
               index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
               #real_income = (Wage[co1] * Skill_level1[skill_it,co1] + price[co2] * np.minimum((1 + r[co2]) * cap_a_gridchoice,cap_a_gridchoice_lim) - price[co2] * a_prime_gridchoice + price[0]*
               #              np.minimum(1/Bond * m_gridchoice,cap_m_gridchoice_lim) - price[0]*m_prime_gridchoice )/price[co1]
               real_income = (Wage[co1] * Skill_level1[skill_it,co1] + price[co2] * (1 + r[co2]) * cap_a_gridchoice - price[co2] * a_prime_gridchoice + price[0]*
                              1/Bond * m_gridchoice - price[0]*m_prime_gridchoice+ Profits[co2] / Czship[co2] )/price[co1]
               Cons_NOadj[:,:,index_tod] = (1 - Adjust_mat[:,:,co2]) * util(real_income) + Adjust_mat[:,:,co2] *Cons_NOadj[:,:,index_tod]
               Cons_adj[:,:,index_tod] = Adjust_mat[:,:,co2] * util(real_income - AdjustCost_applied/(price[co1]) - AdjustCost_applied_quad * income_cost * Adjust_mat[:,:,co2]) + (1 - Adjust_mat[:,:,co2]) * Cons_adj[:,:,index_tod]

    # get a good guess from the simple model:

#    # Workers problem
#    coeff_e = 0* np.ones((illiquid_param_asset[0],No_countries**2 * Skill_level_no))
#    V_val = coeff_e.copy()
#    Phi_prime_store = np.zeros((illiquid_param_asset[0],illiquid_param_asset[0],No_countries**2 * Skill_level_no))
#    #Value function approximation
#    conv = 10
#    iterate = 0
#    while (conv>val_tol):
#        for co2 in range(No_countries):
#            for co1 in range(No_countries):
#                for skill_it in range(Skill_level_no):
#                    index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
#                    real_income = (Wage[co1] * Skill_level1[skill_it,co1] + 
#                           price[co2] * (1 + r[co2]) * illiquid_asset_grid1 * remmittance_savings_rate
#                           + 
#                           price[co1] * (1 + r[co1]) * illiquid_asset_grid1 * (1 - remmittance_savings_rate)
#                          )/price[co1]
#                    
#                    rel_price = (price[co1]*(1-remmittance_savings_rate)+price[co2]*remmittance_savings_rate) / price[co1]
#                    upper_bound_applied = np.minimum(real_income/rel_price,illiquid_upper_bound[:illiquid_param_asset[0],None])
#                    fut_asset1, V_val_temp = ip.goldenx(current_util_simple,illiquid_lower_bound[:illiquid_param_asset[0],None],upper_bound_applied,max_tol,real_income,coeff_e[
#                        :,index_tod],rel_price)
#                    Phi_prime_store[:,:,index_tod] = ip.spli_basex(illiquid_param_asset,fut_asset1.flatten(),deg = degree,knots = knots_illiquid)
#                    #Phi_prime_store[:,:,index_tod] = ip.spli_basex(param_asset,fut_asset1.flatten(),deg = degree)
#                    V_val[:,np.newaxis,index_tod] = V_val_temp
#                    #coeff_next[:,index_tod] = np.linalg.solve(Phi,V_val_temp)[:,0]
#        V_bar_temp = np.zeros((illiquid_param_asset[0],No_countries**2 * Skill_level_no,No_countries))
#        for co in range(No_countries):
#            V_bar_temp[:,:,co] = V_val @ Transition_Matrix_stacked1[:,:,co].T - Util_cost_matrix[:,:illiquid_param_asset[0],co].T
#        index_max = V_bar_temp.argmax(axis =2)
#        V_bar_max = ((index_max == 0 * np.ones(index_max.shape)) * (V_val @ Transition_Matrix_stacked1[:,:,0].T 
#                                                                        - Util_cost_matrix[:,:illiquid_param_asset[0],0].T))
#        for co in range(1,No_countries):
#            V_bar_max = V_bar_max + ((index_max == co * np.ones(index_max.shape)) * (V_val @ Transition_Matrix_stacked1[:,:,co].T 
#                                                                        - Util_cost_matrix[:,:illiquid_param_asset[0],co].T))
#        V_bar = freedom_fairy *V_bar_max + (1 - freedom_fairy) * V_val @ Transition_Matrix_stacked_stay.T
#
#        #coeff, conv1 = Bellman_iter(coeff,coeff_next,dampen)
#        #coeff_e, conv = Bellman_iter(coeff_e,coeff_next_e,dampen,V_bar,index_max)
#        #coeff_e, conv = Bellman_iter2(coeff_e,coeff_next_e,dampen,V_bar)
#        #coeff_e, conv = Bellman_iter2(coeff_e,dampen,V_bar)
#        #coeff_e, conv = Newton_iter2(coeff_e,dampen,V_bar,index_max,Phi,Phi_prime_store)
#        if(iterate < fastcoeff):
#            coeff_e, conv = Bellman_iter(coeff_e,dampen,V_bar)
#        else:
#            coeff_e, conv = Newton_iter(coeff_e,dampen,V_bar,index_max,Phi,Phi_prime_store)
#        if(conv > 10000):
#            conv = 0
#            print('Really bad guess - Bellman iteration does not converge',xsol)
#        iterate = iterate +1
#        print(conv)
#    
    
    # Workers problem
    #coeff_e = np.tile(coeff_e,(liquid_param_asset[0],1)) #
    coeff_e =0*np.ones((AssetMultiplied,No_countries**2 * Skill_level_no))
    #coeff2 = np.loadtxt(open("coeffe.csv", "rb"), delimiter=" ")
    #coeff_e=coeff2
    V_val = coeff_e.copy()
    
    #V_bar_prev = np.zeros((illiquid_param_asset[0]*liquid_param_asset[0],No_countries**2 * Skill_level_no))
    
    #Phi_prime_store = np.zeros((param_asset[0],param_asset[0],No_countries**2 * Skill_level_no))
    #Value function approximation
    conv = 10
    #conv2=10
    #fastcoeff = 2
    #dampen = 0.1
    iterate = 0
    arg_instances=[]
    for co2 in range(No_countries):
        for co1 in range(No_countries):
           for skill_it in range(Skill_level_no):
               arg_instances.append((skill_it,co1,co2))
    Phi_prime_store_m = np.zeros((AssetMultiplied,liquid_param_asset[0],No_countries**2 * Skill_level_no))
    Phi_prime_store_a = np.zeros((AssetMultiplied,illiquid_param_asset[0],No_countries**2 * Skill_level_no))
    while (conv>val_tol):
        V_val = 0*coeff_e.copy()
        if (iterate < fastcoeff):
            if __name__ == "__main__":
                number_processes = 4
                prod_x=partial(Multiprocessfile_gridsearch.LoopFunc,gamma=gamma,Skill_level_no=Skill_level_no,Beta=Beta,degree=degree,No_countries=No_countries,
                               coeff_e=coeff_e,liquid_param_asset=liquid_param_asset,illiquid_param_asset=illiquid_param_asset,liquid_lower_bound=liquid_lower_bound
                               ,liquid_upper_bound=liquid_upper_bound,max_tol=max_tol,Wage=Wage,Skill_level1=Skill_level1,price=price,AdjustCost=AdjustCost,
                               Bond=Bond,illiquid_upper_bound=illiquid_upper_bound,illiquid_lower_bound=illiquid_lower_bound,CapLimiter=CapLimiter,r=r,s=s,
                               AdjustCost_quad=AdjustCost_quad,AdjustCost_curve = AdjustCost_curve,knots_liquid = knots_liquid,knots_illiquid =knots_illiquid,
                               Cons_NOadj = Cons_NOadj,Cons_adj = Cons_adj,Phi_fine = Phi_fine,AssetMultiplied = AssetMultiplied,s_fine=s_fine,
                               fine_grid_no=fine_grid_no,illiquid_param_asset_fine=illiquid_param_asset_fine, Profits = Profits,Czship=Czship)
                pool = multiprocessing.Pool(number_processes)
    
                results =[]
                results = pool.map_async(prod_x, arg_instances)
                ABC = results.get()
                pool.close()
                pool.join()
            for i in range(len(arg_instances)):
                V_val[:,i] =  ABC[i]
        else:
            if __name__ == "__main__":
                number_processes = 4
                prod_x=partial(Multiprocessfile_Newton_gridsearch.LoopFunc,gamma=gamma,Skill_level_no=Skill_level_no,Beta=Beta,degree=degree,No_countries=No_countries,
                               coeff_e=coeff_e,liquid_param_asset=liquid_param_asset,illiquid_param_asset=illiquid_param_asset,liquid_lower_bound=liquid_lower_bound
                               ,liquid_upper_bound=liquid_upper_bound,max_tol=max_tol,Wage=Wage,Skill_level1=Skill_level1,price=price,AdjustCost=AdjustCost,
                               Bond=Bond,illiquid_upper_bound=illiquid_upper_bound,illiquid_lower_bound=illiquid_lower_bound,CapLimiter=CapLimiter,r=r,s=s,
                               AdjustCost_quad=AdjustCost_quad,AdjustCost_curve = AdjustCost_curve,knots_liquid = knots_liquid,knots_illiquid =knots_illiquid,
                               Cons_NOadj = Cons_NOadj,Cons_adj = Cons_adj,Phi_fine = Phi_fine,AssetMultiplied = AssetMultiplied,conv = conv,s_fine=s_fine,
                               fine_grid_no=fine_grid_no,illiquid_param_asset_fine=illiquid_param_asset_fine, Profits = Profits,Czship=Czship)
                pool = multiprocessing.Pool(number_processes)
                results =[]
                results = pool.map_async(prod_x, arg_instances)
                ABC = results.get()
    
                pool.close()
                pool.join()
            for i in range(len(arg_instances)):
                V_val[:,i] =  ABC[i][0]
                Phi_prime_store_a[:,:,i] = ABC[i][1]
                Phi_prime_store_m[:,:,i] = ABC[i][2]
                
        V_bar_temp = np.zeros((AssetMultiplied,No_countries**2 * Skill_level_no,No_countries))
        for co in range(No_countries):
            V_bar_temp[:,:,co] = V_val @ Transition_Matrix_stacked1[:,:,co].T - Util_cost_matrix[:,:,co].T
        index_max = V_bar_temp.argmax(axis =2)
        V_bar_max = ((index_max == 0 * np.ones(index_max.shape)) * (V_val @ Transition_Matrix_stacked1[:,:,0].T 
                                                                        - Util_cost_matrix[:,:,0].T))
        for co in range(1,No_countries):
            V_bar_max = V_bar_max + ((index_max == co * np.ones(index_max.shape)) * (V_val @ Transition_Matrix_stacked1[:,:,co].T 
                                                                        - Util_cost_matrix[:,:,co].T))
        V_bar = freedom_fairy *V_bar_max + (1 - freedom_fairy) * V_val @ Transition_Matrix_stacked_stay.T
        
        if (iterate < fastcoeff):
            coeff_e, conv = Bellman_iter2(coeff_e,dampen,V_bar)
        else:
            #plt.plot(V_bar[:,0])
            coeff_e, conv = Newton_iter2(coeff_e,dampen,V_bar,index_max,Phi,Phi_prime_store_m,Phi_prime_store_a)
        if(iterate >5000):
            conv = 0
            print('Really bad guess - Bellman iteration does not converge',xsol)
        #plt.plot(V_bar)
        #conv2 = np.max(np.absolute(V_bar-V_bar_prev))
        #V_bar_prev = V_bar.copy()
        print(conv)
        iterate = iterate+1
    if (conv == 0  ):
        residual = 1e+15 * xsol
    else:
            # Gridsearch:
    
        Adjust_mat_fine = np.zeros((fine_grid_no,fine_grid_no,No_countries))
        for ii in range(No_countries):
            Adjust_mat_fine[:,:,ii] = cap_a_gridchoice_fine * (1 + (r[ii])) > a_prime_gridchoice_fine
        Cons_NOadj_fine =-10000 * np.ones((fine_grid_no,fine_grid_no,No_countries**2 * Skill_level_no))
        Cons_adj_fine = -10000 * np.ones((fine_grid_no,fine_grid_no,No_countries**2 * Skill_level_no)) 
        for co2 in range(No_countries):
            for co1 in range(No_countries):
                AdjustCost_applied = price[co2]*AdjustCost[co1 + co2 * No_countries]
                AdjustCost_applied_quad = price[co2]/(price[co1])*AdjustCost_quad[co1 + co2 * No_countries]
                for skill_it in range(Skill_level_no):
                   index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
                   #real_income_fine = (Wage[co1] * Skill_level1[skill_it,co1] + price[co2] * np.minimum((1 + r[co2]) * cap_a_gridchoice_fine,cap_a_gridchoice_lim_fine) - price[co2] * a_prime_gridchoice_fine + price[0]*
                   #               np.minimum(1/Bond * m_gridchoice_fine,cap_m_gridchoice_lim_fine) - price[0]*m_prime_gridchoice_fine )/price[co1]
                   real_income_fine = (Wage[co1] * Skill_level1[skill_it,co1] + price[co2] * (1 + r[co2]) * cap_a_gridchoice_fine - price[co2] * a_prime_gridchoice_fine + price[0]*
                                       1/Bond * m_gridchoice_fine - price[0]*m_prime_gridchoice_fine+ Profits[co2] / Czship[co2] )/price[co1]
                   Cons_NOadj_fine[:,:,index_tod] = (1 - Adjust_mat_fine[:,:,co2]) * util(real_income_fine) + Adjust_mat_fine[:,:,co2] *Cons_NOadj_fine[:,:,index_tod]
                   Cons_adj_fine[:,:,index_tod] = Adjust_mat_fine[:,:,co2] * util(real_income_fine - AdjustCost_applied/(price[co1]) - AdjustCost_applied_quad * income_cost_fine * Adjust_mat_fine[:,:,co2]) + (1 - Adjust_mat_fine[:,:,co2]) * Cons_adj_fine[:,:,index_tod]
        Q_store = np.zeros((fine_grid_no,fine_grid_no,No_countries**2 * Skill_level_no))
        V_val_fine = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no))
        cons_fine = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no))
        V_val = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no))
        a_prime_fine = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no))
        m_prime_fine = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no))
        real_AdjustCost_payed = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no))
        index_adjust = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no))
        if __name__ == "__main__":
            number_processes = 4
            prod_x=partial(Multiprocessfile_aggregates_gridsearch.LoopFunc,gamma=gamma,Skill_level_no=Skill_level_no,Beta=Beta,degree=degree,No_countries=No_countries,
                           coeff_e =coeff_e,liquid_param_asset=liquid_param_asset,illiquid_param_asset=illiquid_param_asset,liquid_lower_bound_fine=liquid_lower_bound_fine
                           ,liquid_upper_bound_fine=liquid_upper_bound_fine,max_tol=max_tol,Wage=Wage,Skill_level1=Skill_level1,price=price,AdjustCost=AdjustCost,
                           Bond=Bond,illiquid_upper_bound_fine=illiquid_upper_bound_fine,illiquid_lower_bound_fine=illiquid_lower_bound_fine,CapLimiter=CapLimiter,r=r,s_fine=s_fine,AdjustCost_quad=AdjustCost_quad,
                           liquid_param_asset_fine = liquid_param_asset_fine,illiquid_param_asset_fine = illiquid_param_asset_fine,AdjustCost_curve = AdjustCost_curve,
                           knots_liquid = knots_liquid,knots_illiquid =knots_illiquid,Cons_NOadj_fine = Cons_NOadj_fine,
                           Cons_adj_fine = Cons_adj_fine,Phi_fine = Phi_fine,conv = conv,fine_grid_no = fine_grid_no,liquid_asset_grid_fine = liquid_asset_grid_fine,
                           illiquid_asset_grid_fine = illiquid_asset_grid_fine, Profits = Profits,Czship=Czship)
            pool = multiprocessing.Pool(number_processes)
        
            results =[]
            results = pool.map_async(prod_x, arg_instances)
            ABC = results.get()
            pool.close()
            pool.join()
        for i in range(len(arg_instances)):
            V_val[:,i] =  ABC[i][0]
            Q_store[:,:,i] =ABC[i][1]
            cons_fine[:,i] = ABC[i][2]
            a_prime_fine[:,i] = ABC[i][3]
            m_prime_fine[:,i] = ABC[i][4]
            real_AdjustCost_payed[:,i] = ABC[i][5]
            index_adjust[:,i] = ABC[i][6]
        
        V_bar_temp = np.zeros((fine_grid_no,No_countries**2 * Skill_level_no,No_countries))
        for co in range(No_countries):
            V_bar_temp[:,:,co] = V_val_fine @ Transition_Matrix_stacked1[:,:,co].T - Util_cost_matrix_fine[:,:,co].T
        
        index_max = V_bar_temp.argmax(axis =2)
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
        stat_distr1 = stat_distr_eigen_sparse(BigT)

        print(loc_selector @ stat_distr1)
        L_effective_trans = loc_selector @ (stat_distr1 * skill_level_tile)
        cons_fine1 = cons_fine.reshape((fine_grid_no*No_countries**2 * Skill_level_no,1), order='F')
        a_prime_fine1 = a_prime_fine.reshape((fine_grid_no*No_countries**2 * Skill_level_no,1), order='F')
        m_prime_fine1 = m_prime_fine.reshape((fine_grid_no*No_countries**2 * Skill_level_no,1), order='F')
        K_trans = cit_selector @ (stat_distr1 * a_prime_fine1)
        M_trans = loc_selector @ (stat_distr1 * m_prime_fine1)
        spending = loc_selector @ (stat_distr1 * cons_fine1)
        #cit_selector @ (stat_distr1 * a_prime_fine1)
        residual[:No_countries] =  L - L_effective_trans.T  #L_effective - L_effective_trans.T
        residual[No_countries:(2 * No_countries)] =  K - K_trans.T
        residual[(2 * No_countries):(3 * No_countries - 1)] =  (spending.flatten() + delta * K + Profits/price  - production.flatten())[1:] #+ Profits/No_countries 
        residual[(3 * No_countries - 1)] = M_trans.sum()
        print(residual,xsol)
    return residual
#xsol = np.array([-100, 0.51340516,  0.98525969,  1.20269589,  1.98955914,  0.39574654])
#xsol = np.array( [-100, 0.61515149,  0.93623952, -1.01932491,  1.54998848,  0.35362593])#Working for tau=0
#from scipy.optimize import linprog
#xsol = np. array([  5.57197801e-01,   8.78359165e-02,  -4.97162565e-01,1.04113621e+00,   1.91026974e-02,  -9.99999408e+01])#Working for tau=1000, no fixed
#xsol = np. array([ 0.54056099,  0.08418227 ,-0.18278302 , 1.31323422 , 0.02134582 ,-3.01863998]) ##Working for tau=1000, with fixed tau = 0.03
xsol=np.array([ 0.80510308 , 0.52087383 , 2.21853292 , 3.06566437,  0.17093773, -1.12827824])
#xsol = np.array([ 0.57274441,  0.07523245, -0.23441563 , 1.27451142,  0.03739227,-100])#Working for tau=0, with fixed tau = 0.03




xsol=np.array([ 0.67450235,  0.30049569,  0.95152533,  13.33352873,  0.09863678, -1.12827824])
#x_root1 = optimize.root(stst_resid_faster,xsol,options={ 'ftol': root_tol,'disp': True} ,method = 'df-sane')
def bisection(x_start = xsol,root_tol = root_tol,maxiter = 1000,resid_dampen = 0.05):
    xsol = x_start
    resid = stst_resid_faster(xsol)
    tol = (np.sum(resid**2))**(1/2)
    iterate1 = 0
    while (tol >root_tol and iterate1< maxiter):
        iterate1 = iterate1 +1
        xsol = xsol + np.minimum(tol,resid_dampen)  * resid * np.abs(resid)
        resid = stst_resid_faster(xsol)
        tol = np.sum(resid**2)**(1/2)#/iterate  
        print(tol,iterate1)
    print(np.sum(resid**2)**(1/2))
    return xsol

#Simulations:
    
def simulation(simu_size = 100):
    income_path = np.ones((simu_size,1)) 
    location_path = 0 * np.ones((simu_size,1))
    citizenship_path = 0 *np.ones((simu_size,1))
    liquid_path = 0 *np.ones((simu_size,1))
    illiquid_path = 0 *np.ones((simu_size,1))
    for t in range(simu_size-1):
        skill_it = int(income_path[t,0])
        co1 = int(location_path[t,0])
        co2 = int(citizenship_path[t,0])
        index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
        income_tmp = Skill_level1[skill_it,co1]
        illiquid_weight = ip.spli_basex(illiquid_param_asset,illiquid_path[t,0,None],deg = degree,knots = illiquid_asset_grid_fine )
        liquid_weight = ip.spli_basex(liquid_param_asset,liquid_path[t,0,None],deg = degree,knots = liquid_asset_grid_fine)
        weights_combined = ip.dprod(liquid_weight,illiquid_weight)
        nonzero_weight = (weights_combined > 1e-10)
        state_approx = s_fine[nonzero_weight[0,:] ==1,:]
        a_prime_approx = a_prime_fine[nonzero_weight[0,:] ==1,index_tod]
        m_prime_approx = m_prime_fine[nonzero_weight[0,:] ==1,index_tod]
        m_next = np.sum(weights_combined[0,nonzero_weight[0,:] ==1] * m_prime_approx)
        a_next = np.sum(weights_combined[0,nonzero_weight[0,:] ==1] * a_prime_approx)
        location_path[t+1,0] = np.min(index_max[nonzero_weight[0,:] ==1,index_tod])
        illiquid_path[t+1,0] = a_next
        liquid_path[t+1,0] = m_next
    plt.plot(illiquid_path)
    plt.plot(liquid_path)
    plt.plot(location_path)
    return 1