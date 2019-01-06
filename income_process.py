# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:34:00 2018

@author: laszl
"""
def income_process(No_countries = 2 ,Skill_level_no = 2,grid_rouwen = 2,unemp = 1):
    ##Estimation of the income process:
    from Rouwenhorst import Rouwenhorst
    import numpy as np
    import pandas as pd
    #unemp included
    rho_z = 0.79
    lognormal_distr_data = pd.read_csv('Migration Datawork\HFCS Datawork\DistParameters.csv')
    skill_data = pd.read_csv('Migration Datawork\HFCS Datawork\SkillDist.csv')
    
    skill_mat = np.zeros((Skill_level_no,No_countries*Skill_level_no))
    if unemp == 1:
        prod_mat = np.zeros((Skill_level_no*(grid_rouwen + unemp),No_countries*Skill_level_no*(grid_rouwen + unemp))) 
        Skill_level1 = np.zeros((Skill_level_no*(grid_rouwen + unemp),No_countries))
    else:
        prod_mat = np.zeros((Skill_level_no*grid_rouwen,No_countries*Skill_level_no*grid_rouwen))
        Skill_level1 = np.zeros(((Skill_level_no)*grid_rouwen,No_countries))
    high_skill_perc = skill_data.iloc[2:4,3].values
    low_skill_persist = 0.9 # common across countries  - first sub_matrix is for Poland
    skill_mat[0,0] = 1 - (1 - high_skill_perc[1])/high_skill_perc[1]* (1 - low_skill_persist)
    skill_mat[0,1] = 1 - skill_mat[0,0]
    skill_mat[0,2] = 1 - (1 - high_skill_perc[0])/high_skill_perc[0]*(1 - low_skill_persist)
    skill_mat[0,3] = 1 - skill_mat[0,2]
    skill_mat[1,0] = 1- low_skill_persist
    skill_mat[1,1] = low_skill_persist
    skill_mat[1,2] = 1- low_skill_persist
    skill_mat[1,3] = low_skill_persist
    #Unemployment data:
    sep_rate = 0.02 * np.ones((No_countries*Skill_level_no,1))
    sep_rate[0] = 0.03
    finding_rate = 0.2 * np.ones((No_countries*Skill_level_no,1))
    unemp_inc = 0.1
    #print(sep_rate / (sep_rate + finding_rate))
    
    for skill_it in range(Skill_level_no):
        for co1 in range(No_countries):
            co2 = (No_countries - co1-1)
            skill_it2 = (Skill_level_no - skill_it-1)
            i = co2  + skill_it2* No_countries
            #print(i,skill_it,co1)
            T,zgrid = Rouwenhorst(rho_z,lognormal_distr_data['LogNormal StD'][i]**2, grid_rouwen )
            zgrid = lognormal_distr_data['LogNormal Mean'][i]/(1 - rho_z)  - 1 + np.reshape(zgrid,(grid_rouwen,1))
            # Add unemployment state with earnings of 10% for each skill level
            if unemp == 1:
                T1 = np.zeros((grid_rouwen +unemp,grid_rouwen +unemp ))
                zgrid1 = np.ones((grid_rouwen + unemp,1))
                zgrid1[:grid_rouwen,0] = zgrid[:,0]
                zgrid1[grid_rouwen,0] =  zgrid[:,0].mean()
                T1[:grid_rouwen,:grid_rouwen] = T*(1 - sep_rate[co1 + skill_it2 *No_countries ])
                T1[:grid_rouwen,grid_rouwen:] = np.ones((grid_rouwen,1))*sep_rate[co1 + skill_it2 *No_countries ]
                T1[grid_rouwen:,grid_rouwen:] = 1 - finding_rate[co1 + skill_it2 *No_countries ]
                T1[grid_rouwen:,:grid_rouwen] = finding_rate[co1 + skill_it2 *No_countries ]/grid_rouwen
                prod_mat[(skill_it * (grid_rouwen + unemp)):((skill_it+1) * (grid_rouwen + unemp)), (co1 * Skill_level_no*(grid_rouwen + unemp)
                         +skill_it * (grid_rouwen + unemp)):(co1 * Skill_level_no*(grid_rouwen + unemp) +(skill_it+1) * (grid_rouwen + unemp)
                          )] = T1*skill_mat[skill_it,skill_it +co1 * Skill_level_no]
                prod_mat[(skill_it * (grid_rouwen + unemp)):((skill_it+1) * (grid_rouwen + unemp)), (co1 * Skill_level_no*(grid_rouwen + unemp
                         )+skill_it2 * (grid_rouwen + unemp)):(co1 * Skill_level_no*(grid_rouwen + unemp) +(skill_it2+1) * (grid_rouwen + unemp)
                          )] = T1*skill_mat[skill_it,skill_it2 +co1 * Skill_level_no]
                Skill_level1[(skill_it*(grid_rouwen + unemp)):((skill_it+1)*(grid_rouwen + unemp)),co1] = zgrid1[:,0]                
            #print(zgrid)
            else:
                prod_mat[(skill_it * grid_rouwen):((skill_it+1) * grid_rouwen), (co1 * Skill_level_no*grid_rouwen +skill_it * grid_rouwen):(co1 * Skill_level_no*grid_rouwen +(skill_it+1) * grid_rouwen
                          )] = T*skill_mat[skill_it,skill_it +co1 * Skill_level_no]
                prod_mat[(skill_it * grid_rouwen):((skill_it+1) * grid_rouwen), (co1 * Skill_level_no*grid_rouwen +skill_it2 * grid_rouwen):(co1 * Skill_level_no*grid_rouwen +(skill_it2+1) * grid_rouwen
                          )] = T*skill_mat[skill_it,skill_it2 +co1 * Skill_level_no]
                Skill_level1[(skill_it*grid_rouwen):((skill_it+1)*grid_rouwen),co1] = zgrid[:,0]
    prod_norm_level = Skill_level1.max()
    Skill_level1 = np.exp(Skill_level1)/np.exp(prod_norm_level)
    #Skill_level1 = Skill_level1/prod_norm_level
    if unemp == 1:
        for ii in range(1,Skill_level_no+1):
            #print(ii)
            Skill_level1[ii*(grid_rouwen+unemp)-1,:] = unemp_inc * Skill_level1[((ii-1)*(grid_rouwen+unemp)):((ii)*(grid_rouwen+unemp)-1),:].mean(0)
    Skill_Matrix_stacked = prod_mat.copy() 
    Prod_level = Skill_level_no*(grid_rouwen + unemp)
    Transition_Matrix = np.eye(2*Prod_level,Prod_level)
    Tranistion_High = 1- 0.34
    Transition_Low = 1.
    
    
#    Transition_Matrix[6,0] = Tranistion_High
#    Transition_Matrix[6,2] = (1-Tranistion_High)
#    Transition_Matrix[7,1] = Tranistion_High
#    Transition_Matrix[7,2] = (1-Tranistion_High)
#    Transition_Matrix[8,2] = 1
    Transition_Matrix[6,:] = 1/(Skill_level_no * (grid_rouwen + unemp))
    Transition_Matrix[7,:] = 1/(Skill_level_no * (grid_rouwen + unemp))
    Transition_Matrix[8,:] = 1/(Skill_level_no * (grid_rouwen + unemp))
    Transition_Matrix[9,3] = Transition_Low
    Transition_Matrix[9,0] = (1-Transition_Low)
    Transition_Matrix[10,4] = Transition_Low
    Transition_Matrix[10,1] = (1-Transition_Low)
    Transition_Matrix[11,5] = Transition_Low
    Transition_Matrix[11,2] = (1-Transition_Low)
    return Skill_level1,Skill_Matrix_stacked,Prod_level,Transition_Matrix#,prod_norm_level