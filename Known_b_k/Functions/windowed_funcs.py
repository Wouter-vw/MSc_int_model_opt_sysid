import numpy as np
from tools import control_design
from tvopt import costs, sets
import matplotlib.pyplot as plt
from SYSID import RLS_func

def window_error(b_hat, e_k_RLS, b, k, order, win_size):
    if k <= win_size:
        return 100  # No computation if k is within window size

    else: 
        win_err = e_k_RLS + sum(np.linalg.norm(
        b[:, k].reshape((len(b), 1)) + np.array([b[:, k - i - 1] for i in range(order)]).T @ b_hat,
        ord=1) for k in range(k - 1, k - win_size, -1))

    return win_err

# Online gradient with compensation, designed using control theory
def online_gradient_control_sys_id_ARM_window(f, b, lambda_lims, delta, f_factor, normalized_signal, step, win_size, x_0=0):
    
    ## Initialize needed values (More than just RLS due to added complexity)
    x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    order = 1
    dim = 0
    maxy = int(np.log(f.time.num_samples)**(1 + delta))
    c = np.ones(order)
    y = [np.zeros(f.dom.shape) for _ in range(order)]
    b_hat = np.ones((order,1))
    e_k_norm_best = 1e3
    e_k_norm = 1e2
    e_k_RLS = 1e3
    e_k_init = np.ones_like(b[:,0])*1e3
    P = np.eye(order) * 1  
    c = np.ones(order+1)
    n = 0
    test_coeffs_prev = np.zeros(1)
    test_coeffs = np.zeros(1)
    k_best = None
    b_normed = np.empty_like(b)
    all_coeffs_list = []
    used_index = []
    threshold_reached = False
    norm_list = []
    
    for k in range(f.time.num_samples):
        ## Here we grow the order based on log(k)^(1+delta)
        ## We must add some value error cases to let it run properly
        order_prev = order
        try:
            if k <= 3:
                raise ValueError("k must be greater than 0")
            
            growth = np.log(k)**(1 + delta)
            if np.isinf(growth):
                raise OverflowError("Growth value is too large")
            order = int(growth)
    
        except (ValueError, OverflowError) as e:
            None
    
        ## Make sure to extend our b_hat and P when the order grows (As done in ARM(inf))
        if order != order_prev:
            b_hat = np.vstack((b_hat, [[1]]))
            P = np.pad(P, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            P[-1, -1] = 1

    ## If we want a normalized signal, for example if we have unbounded growth, set flag as true    
        if normalized_signal == True:     
            if k > 2:
                ## Run RLS on our normalized signal and append to history list. Also calculate our initial error
                ## By adding the error of the RLS and the error at a given point best generalization results are seen!
                b_normed = (b[:,:k]  - np.mean(b[:,:k], axis = 1, keepdims=True))/np.max(b[:,:k], axis = 1, keepdims=True)
                b_hat, P, e_k_RLS = RLS_func(b_hat, P, f_factor, k-2, b_normed, order)
                all_coeffs_list.append(np.append(b_hat[::-1], 1.0))
                e_k_norm = window_error(b_hat, e_k_RLS, b_normed, k-2, order, win_size=win_size)

        else:
                ## Run RLS and append to history list. Also calculate our initial error
                ## By adding the error of the RLS and the error at a given point best generalization results are seen!
                b_hat, P, e_k_RLS = RLS_func(b_hat, P, f_factor, k, b, order)
                all_coeffs_list.append(np.append(b_hat[::-1], 1.0))
                e_k_norm = window_error(b_hat, e_k_RLS, b, k, order, win_size=win_size)
        
        norm_list.append(e_k_norm)
    
        ### If are norm improves we want to keep the computed coefficients
        ### We also calculate ahead to see if the order will grow within the next 5 timesteps
        if e_k_norm < e_k_norm_best and order != 1:
            e_k_norm_best = e_k_norm
            b_hat_best = b_hat
            k_best = k
            order_k_best_5 = int(np.log(k_best + 5) ** (1 + delta))
        
        ### if we have improved our error and we haven't had an improvement in the past fivetime steps, or 
        ### the order will grow within the next five timesteps, compute our estimated coefficients
        if k_best is not None and (k == k_best + 5 or order != order_k_best_5) and order > 1:    
            threshold_reached = True
            test_coeffs = np.append(b_hat_best[::-1], 1.0)  # Reverse and append 1.0
    
    ## If we haven't reached our threshold use online gradient (Nicola's code)
        if threshold_reached == False:
            y_uncontrolled = x[...,k]
        
            y_uncontrolled = y_uncontrolled - step*f.gradient(y_uncontrolled, k*f.time.t_s)
            
            x[...,k+1] = y_uncontrolled
    
        ## Once we have reached a stable solution
        ## Run the original algorithm with our calculated coefficients, keeping track of how often we do it
        if threshold_reached == True:
            if test_coeffs.shape != test_coeffs_prev.shape or np.any(test_coeffs != test_coeffs_prev):                
                if order != order_k_best_5:
                    used_index.append(k)
                else: 
                    used_index.append(k - 5)
                n = n + 1
                c = control_design(test_coeffs, lambda_lims)
                test_coeffs_prev = test_coeffs
                if order != dim:
                    y = [np.zeros(f.dom.shape) for _ in range(order)]
                dim = order
            
            e = - f.gradient(x[...,k], k*f.time.t_s)
            
            y = y[1:] + [-sum([test_coeffs[i]*y[i]/test_coeffs[dim] for i in range(dim)]) + e/test_coeffs[dim]]
            
            x[...,k+1] = sum([c[i]*y[i] for i in range(dim)])
            
    ## Printing extra debugging info ##
    print(f"We have had to solve the LMIs {n} times.")
    print(f"The minimum error norm is {e_k_norm_best}")
    return x, test_coeffs, all_coeffs_list, used_index, norm_list