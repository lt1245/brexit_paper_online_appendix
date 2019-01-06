import numpy as np
from poly_base import interpolate as ip
from scipy import sparse
def LoopFunc(x,gamma,Skill_level_no,Beta,degree,No_countries,coeff_e,liquid_param_asset,illiquid_param_asset,liquid_lower_bound,liquid_upper_bound,
             max_tol,Wage,Skill_level1,price,AdjustCost,Bond,illiquid_upper_bound,illiquid_lower_bound,CapLimiter,r,s,AdjustCost_quad,AdjustCost_curve,
             knots_liquid,knots_illiquid,Cons_NOadj,Cons_adj,Phi_fine,AssetMultiplied,conv,s_fine,fine_grid_no,illiquid_param_asset_fine, Profits,Czship):
    #def util(cons):
    #    return cons**(1 - gamma) / (1 - gamma)

    skill_it=int(x[0])
    co1=int(x[1])
    co2=int(x[2])
    checkerror_value = -5000
    index_tod = skill_it + co1 * Skill_level_no + co2 * No_countries * Skill_level_no
    Cap_adj_lim = s[:,0]*(1+r[co2])#np.minimum(s[:,0]*(1+r[co2]),illiquid_upper_bound)#
    # Gridsearch
    EV_grid = Beta *np.tile((Phi_fine @ coeff_e[:,index_tod,None]).T,(AssetMultiplied,1))
    V_adj = Cons_adj[:,:,index_tod] + EV_grid
    adjust_state_index = V_adj.argmax(1)
    V_NOadj = Cons_NOadj[:,:,index_tod] + EV_grid
    NOadjust_state_index = V_NOadj.argmax(1)
    center_grid_adj = s_fine[adjust_state_index,:]
    center_grid_NOadj = s_fine[NOadjust_state_index,:]


    #Bounds for goldenx    
    if conv <0.0000000001:
    
        s_min = s_fine.min(0) 
        s_max = s_fine.max(0) 
        above_min_adj = (center_grid_adj > s_min) 
        min_grid_adj = above_min_adj * np.concatenate((s_fine[adjust_state_index-1,0,None],s_fine[adjust_state_index-illiquid_param_asset_fine[0],1,None]),axis =1) + (1 - above_min_adj) *  s_min
        below_max_adj = (center_grid_adj < s_max) 
        max_grid_adj = below_max_adj * np.concatenate((s_fine[np.minimum(adjust_state_index+1,fine_grid_no-1),0,None],s_fine[np.minimum(adjust_state_index+illiquid_param_asset_fine[0],fine_grid_no-1),1,None]),axis =1
                                                      ) + (1 - below_max_adj) *  s_max
        max_grid_adj = np.concatenate((np.minimum(max_grid_adj[:,0,None],Cap_adj_lim[:,None]),max_grid_adj[:,1,None]),axis = 1)
        min_grid_adj = np.minimum(min_grid_adj,max_grid_adj)
        above_min_NOadj = (center_grid_NOadj > s_min) 
        min_grid_NOadj = above_min_NOadj * np.concatenate((s_fine[NOadjust_state_index-1,0,None],s_fine[NOadjust_state_index-illiquid_param_asset_fine[0],1,None]),axis =1) + (1 - above_min_NOadj) *  s_min
        below_max_NOadj = (center_grid_NOadj < s_max) 
        max_grid_NOadj = below_max_NOadj * np.concatenate((s_fine[np.minimum(NOadjust_state_index+1,fine_grid_no-1),0,None],s_fine[np.minimum(NOadjust_state_index+illiquid_param_asset_fine[0],fine_grid_no-1),1,None]),axis =1
                                                          ) + (1 - below_max_NOadj) *  s_max
        min_grid_NOadj = np.concatenate((np.maximum(min_grid_NOadj[:,0,None],Cap_adj_lim[:,None]),min_grid_NOadj[:,1,None]),axis = 1)   
        max_grid_NOadj = np.concatenate((np.maximum(min_grid_NOadj[:,0,None],max_grid_NOadj[:,0,None]),max_grid_NOadj[:,1,None]),axis = 1) 
        # Saving values for future use and to check if they are admissible as bounds for goldenx
        # Min adj
        a_goldenx_min_grid_adj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), above_min_adj[:,0]*(adjust_state_index-1) + (1 - above_min_adj[:,0])*adjust_state_index)), (AssetMultiplied, fine_grid_no))
        V_grid_adj_a_min = (a_goldenx_min_grid_adj.toarray() * V_adj).sum(1)    
        m_goldenx_min_grid_adj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), above_min_adj[:,1]*(adjust_state_index-illiquid_param_asset_fine[0]) + (1 - above_min_adj[:,1])*adjust_state_index)), (AssetMultiplied, fine_grid_no))
        V_grid_adj_m_min = (m_goldenx_min_grid_adj.toarray() * V_adj).sum(1)  
        gridsearch_nonsensical_adjust_a_min = V_grid_adj_a_min < checkerror_value
        gridsearch_nonsensical_adjust_m_min = V_grid_adj_m_min < checkerror_value
        # Max adj
        a_goldenx_max_grid_adj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), below_max_adj[:,0]*(np.minimum(adjust_state_index+1,fine_grid_no-1)) + (1 - below_max_adj[:,0])*adjust_state_index)), (AssetMultiplied, fine_grid_no))
        V_grid_adj_a_max = (a_goldenx_max_grid_adj.toarray() * V_adj).sum(1)    
        m_goldenx_max_grid_adj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), below_max_adj[:,1]*(np.minimum(adjust_state_index+illiquid_param_asset_fine[0],fine_grid_no-1)) + (1 - below_max_adj[:,1])*adjust_state_index)), (AssetMultiplied, fine_grid_no))
        V_grid_adj_m_max = (m_goldenx_max_grid_adj.toarray() * V_adj).sum(1)  
        gridsearch_nonsensical_adjust_a_max = V_grid_adj_a_max < checkerror_value
        gridsearch_nonsensical_adjust_m_max = V_grid_adj_m_max < checkerror_value
        # Min NOadj
        a_goldenx_min_grid_NOadj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), above_min_NOadj[:,0]*(NOadjust_state_index-1) + (1 - above_min_NOadj[:,0])*NOadjust_state_index)), (AssetMultiplied, fine_grid_no))
        V_grid_NOadj_a_min = (a_goldenx_min_grid_NOadj.toarray() * V_NOadj).sum(1)    
        m_goldenx_min_grid_NOadj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), above_min_NOadj[:,1]*(NOadjust_state_index-illiquid_param_asset_fine[0]) + (1 - above_min_NOadj[:,1])*NOadjust_state_index)), (AssetMultiplied, fine_grid_no))
        V_grid_NOadj_m_min = (m_goldenx_min_grid_NOadj.toarray() * V_NOadj).sum(1)
        gridsearch_nonsensical_NOadjust_a_min = V_grid_NOadj_a_min < checkerror_value
        gridsearch_nonsensical_NOadjust_m_min = V_grid_NOadj_m_min < checkerror_value
        # Max NOadj
        a_goldenx_max_grid_NOadj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), below_max_NOadj[:,0]*(np.minimum(NOadjust_state_index+1,fine_grid_no-1)) + (1 - below_max_NOadj[:,0])*NOadjust_state_index)), (AssetMultiplied, fine_grid_no))
        V_grid_NOadj_a_max = (a_goldenx_max_grid_NOadj.toarray() * V_NOadj).sum(1)    
        m_goldenx_max_grid_NOadj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), below_max_NOadj[:,1]*(np.minimum(NOadjust_state_index+illiquid_param_asset_fine[0],fine_grid_no-1)) + (1 - below_max_NOadj[:,1])*NOadjust_state_index)), (AssetMultiplied, fine_grid_no))
        V_grid_NOadj_m_max = (m_goldenx_max_grid_NOadj.toarray() * V_NOadj).sum(1)  
        gridsearch_nonsensical_NOadjust_a_max = V_grid_NOadj_a_max < checkerror_value
        gridsearch_nonsensical_NOadjust_m_max = V_grid_NOadj_m_max < checkerror_value
        #Search For Point
        A_lower_search_adj = (1 - gridsearch_nonsensical_adjust_a_min) * min_grid_adj[:,0] + gridsearch_nonsensical_adjust_a_min * center_grid_adj[:,0]
        A_upper_search_adj = (1 - gridsearch_nonsensical_adjust_a_max) * max_grid_adj[:,0] + gridsearch_nonsensical_adjust_a_max * center_grid_adj[:,0]
        M_lower_search_adj = (1 - gridsearch_nonsensical_adjust_m_min) *min_grid_adj[:,1]+ gridsearch_nonsensical_adjust_m_min * center_grid_adj[:,1]
        M_upper_search_adj = (1 - gridsearch_nonsensical_adjust_m_max) *max_grid_adj[:,1]+ gridsearch_nonsensical_adjust_m_max * center_grid_adj[:,1]
    #    
        A_lower_search_NOadj = (1 - gridsearch_nonsensical_NOadjust_a_min) * min_grid_NOadj[:,0] + gridsearch_nonsensical_NOadjust_a_min * center_grid_NOadj[:,0]
        A_upper_search_NOadj = (1 - gridsearch_nonsensical_NOadjust_a_max) * max_grid_NOadj[:,0] + gridsearch_nonsensical_NOadjust_a_max * center_grid_NOadj[:,0]
        M_lower_search_NOadj = (1 - gridsearch_nonsensical_NOadjust_m_min) *min_grid_NOadj[:,1]+ gridsearch_nonsensical_NOadjust_m_min * center_grid_NOadj[:,1]
        M_upper_search_NOadj = (1 - gridsearch_nonsensical_NOadjust_m_max) *max_grid_NOadj[:,1]+ gridsearch_nonsensical_NOadjust_m_max * center_grid_NOadj[:,1]
        #Fixed Point
    else:
        A_lower_search_adj = center_grid_adj[:,0]
        A_upper_search_adj = center_grid_adj[:,0]
        M_lower_search_adj = center_grid_adj[:,1]
        M_upper_search_adj = center_grid_adj[:,1]   
        A_lower_search_NOadj = center_grid_NOadj[:,0]
        A_upper_search_NOadj = center_grid_NOadj[:,0]
        M_lower_search_NOadj = center_grid_NOadj[:,1]
        M_upper_search_NOadj = center_grid_NOadj[:,1]   
    goldenx_center_grid_adj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), adjust_state_index)), (AssetMultiplied, fine_grid_no))
    V_grid_adj_opt = (goldenx_center_grid_adj.toarray() * V_adj).sum(1)    
    goldenx_center_grid_NOadj = sparse.csr_matrix((np.ones(AssetMultiplied), (np.arange(0,AssetMultiplied,1), NOadjust_state_index)), (AssetMultiplied, fine_grid_no))
    V_grid_NOadj_opt = (goldenx_center_grid_NOadj.toarray() * V_NOadj).sum(1)    
    gridsearch_nonsensical_NOadjust = V_grid_NOadj_opt < checkerror_value
    gridsearch_nonsensical_adjust = V_grid_adj_opt < checkerror_value
    # Value function approximation
    def util(cons):
        res = cons > 0
        res1 = np.absolute(cons)
        return res1**(1 - gamma) / (1 - gamma) * res - (1-res) * (10000000 * res1) 
    def current_util_xy(m_prime,a_prime,income,coeff1,rel_price_a,rel_price_m,Phi_prime_a):
        Phi_prime_m = ip.spli_basex(liquid_param_asset,m_prime,deg = degree,knots = knots_liquid)
        Phi_prime = ip.dprod(Phi_prime_m,Phi_prime_a)
        EV = Beta*(Phi_prime @ coeff1)
        V_fut = util(income - rel_price_a * a_prime- rel_price_m * m_prime) + EV
        return V_fut
    
    def current_util_x(a_prime,income,coeff1,rel_price_a,rel_price_m,quad_cost_param,cap):
        Phi_prime_a = ip.spli_basex(illiquid_param_asset,a_prime,deg = degree,knots = knots_illiquid)
        if quad_cost_param >0:
            income_cost = np.float_power( np.maximum((cap - a_prime),0),AdjustCost_curve)
            liquid_lower_bound_applied=M_lower_search_adj
            upper_bound_applied1=M_upper_search_adj
            quad_cost_param=quad_cost_param-1.
        else:
            income_cost = 0
            liquid_lower_bound_applied=M_lower_search_NOadj
            upper_bound_applied1=M_upper_search_NOadj        
        income1 = income - quad_cost_param *income_cost
        m_prime, V_fut = ip.goldenx(current_util_xy,liquid_lower_bound_applied,upper_bound_applied1,max_tol,a_prime,income1,coeff1,rel_price_a,rel_price_m,Phi_prime_a)
        return V_fut




    AdjustCost_applied = price[co2]*AdjustCost[co1 + co2 * No_countries]
    AdjustCost_applied_quad = price[co2]*AdjustCost_quad[co1 + co2 * No_countries]/(price[co1])
    #real_income = (Wage[co1] * Skill_level1[skill_it,co1] +price[0]*np.minimum(s[:,1]/Bond,liquid_upper_bound) + price[co2]*Cap_adj_lim)/(price[co1]) - AdjustCost_applied/price[co1]
    real_income = (Wage[co1] * Skill_level1[skill_it,co1] +price[0]*s[:,1]/Bond+ price[co2]*Cap_adj_lim+ Profits[co2] / Czship[co2])/(price[co1]) - AdjustCost_applied/price[co1]
    rel_price_a = price[co2]/(price[co1])
    rel_price_m = price[0]/(price[co1])
 
    fut_asset_adjust, V_adjust_temp = ip.goldenx(current_util_x,A_lower_search_adj,A_upper_search_adj,max_tol,real_income,coeff_e[:,index_tod],rel_price_a,rel_price_m,AdjustCost_applied_quad+1.,Cap_adj_lim)
    Phi_prime_a_temp_adjust = ip.spli_basex(illiquid_param_asset,fut_asset_adjust,deg = degree,knots = knots_illiquid)
    real_income_adjust1 = real_income - AdjustCost_applied_quad* np.float_power( np.maximum((Cap_adj_lim - fut_asset_adjust),0),AdjustCost_curve)
    #upper_bound_applied2 = np.minimum((real_income_adjust1 - rel_price_a * fut_asset_adjust) / rel_price_m, liquid_upper_bound) #liquid_upper_bound #np.minimum((real_income - rel_price_a *fut_asset_adjust) / rel_price_m, liquid_upper_bound)        
    m_prime_adjust, V_adjust_temp1 =ip.goldenx(current_util_xy,M_lower_search_adj,M_upper_search_adj,max_tol,fut_asset_adjust,real_income_adjust1,coeff_e[:,index_tod],rel_price_a,rel_price_m,Phi_prime_a_temp_adjust)
    

    #Non-adjust
    #real_income_NOadjust = (Wage[co1] * Skill_level1[skill_it,co1] +price[0]*np.minimum(s[:,1]/Bond,liquid_upper_bound) + price[co2]*Cap_adj_lim)/(price[co1])       
    real_income_NOadjust = (Wage[co1] * Skill_level1[skill_it,co1] +price[0]*s[:,1]/Bond + price[co2]*Cap_adj_lim+ Profits[co2] / Czship[co2])/(price[co1])      
    fut_asset_NOadjust, V_NOadjust_temp = ip.goldenx(current_util_x,A_lower_search_NOadj,A_upper_search_NOadj,max_tol,real_income_NOadjust,coeff_e[:,index_tod],rel_price_a,rel_price_m,0,Cap_adj_lim)
    Phi_prime_a_temp_NOadjust = ip.spli_basex(illiquid_param_asset,fut_asset_NOadjust,deg = degree,knots = knots_illiquid)

    m_prime_NOadjust, V_NOadjust_temp1 =ip.goldenx(current_util_xy,M_lower_search_NOadj,M_upper_search_NOadj,max_tol,fut_asset_NOadjust,real_income_NOadjust,coeff_e[:,index_tod],rel_price_a,rel_price_m,Phi_prime_a_temp_NOadjust)

    V_NOadjust_temp_true = V_NOadjust_temp * (1 - gridsearch_nonsensical_NOadjust) + checkerror_value * gridsearch_nonsensical_NOadjust
    
    V_adjust_temp_true = V_adjust_temp * (1 - gridsearch_nonsensical_adjust) + checkerror_value * gridsearch_nonsensical_adjust
    index_adjust = np.argmax(np.concatenate((V_NOadjust_temp_true[:,None],V_adjust_temp_true[:,None] ),axis=1),axis =1)
    V_val_temp = V_adjust_temp_true * index_adjust + (1 - index_adjust) * V_NOadjust_temp_true

    #V_val_temp = (V_adjust_temp * index_adjust + (1 - index_adjust) * V_NOadjust_temp)
    m_prime_final = m_prime_adjust * index_adjust + (1 - index_adjust) *  m_prime_NOadjust
    a_prime_final =  fut_asset_adjust * index_adjust + (1 - index_adjust) *  fut_asset_NOadjust
    Phi_prime_a_temp = ip.spli_basex(illiquid_param_asset,a_prime_final,deg = degree,knots = knots_illiquid)
    Phi_prime_m_temp = ip.spli_basex(liquid_param_asset,m_prime_final,deg = degree,knots = knots_liquid)
    return V_val_temp,Phi_prime_a_temp,Phi_prime_m_temp