import numpy as np
from tools import control_design
from tvopt import costs, sets
import matplotlib.pyplot as plt
from SYSID import RLS_func, calculate_norm_error
from windowed_funcs import window_error

import numpy as np
from tools import control_design
from tvopt import costs, sets
import matplotlib.pyplot as plt

## This is the function that performs RLS for given k
def RLS_func(b_hat, P, f_factor, k, b, order):
    # Form the regressor vector with the order which is given at k given
    h_k = np.array([b[:, k - i - 1] for i in range(order)])
    
    # Compute the error: e_k = b_k - h_k^T b_hat
    e_k_bef = b[:, k].reshape((len(b),1)) + h_k.T @ b_hat  

    ### Reset the P matrix to preserve stability (This must be done sometimes as the signal may not excite RLS enough!
    if np.linalg.norm(P) > 1e16:
        P = np.eye(order) * 1 
    
    # Update RLS gain (In the case that we encounter a singular matrix we try a potential more stable calculation, in practice has not be activated. 
    try:
        K_k = P @ h_k @ np.linalg.inv(np.eye(len(b)) * f_factor + h_k.T @ P @ h_k)
    except np.linalg.LinAlgError:
        print("Error: Singular matrix encountered.")
        K_k = P @ h_k @ np.linalg.inv(np.eye(len(b)) + h_k.T @ P @ h_k)

    # Update parameter estimate
    b_hat = b_hat - K_k @ e_k_bef
    
    # Calculate the error after update
    e_k = b[:, k].reshape((len(b),1)) + h_k.T @ b_hat  
    
    # Take error norm 
    e_k_norm = np.linalg.norm(e_k,ord=1)
    
    # Update covariance matrix
    P = (P - (K_k @ h_k.T @ P)) / f_factor
    return b_hat, P, e_k_norm


# Online gradient with built in RLS
def online_gradient_control_sys_id(f, b, lambda_lims, order, f_factor, step, different_systems, A_tilde, x_0=0):
    ## Initialize needed values
    ## A lot of variables initialized here, the last lists are used for better tracking of solution evolution
    x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    y = [np.zeros(f.dom.shape) for _ in range(order)]
    b_hat = np.ones((order,1))
    e_k_norm_best = 1e3
    test_coeffs = np.ones(order + 1)
    test_coeffs_prev = np.zeros(order + 1)
    P = np.eye(order)
    c = np.ones(order+1)
    n = 0
    threshold_reached = False
    all_coeffs_list = []
    used_index = []
    norm_list = []
    perturbed_b_all = np.zeros_like(b)
    
    ### This is the main loop for our system ID
    for k in range(f.time.num_samples):
        ## Run sys ID and append the history of RLS ##
        perturbed_b = A_tilde[k] @ x[:, :, k] + b[:, k].reshape(-1, 1)
        #perturbed_b = A_tilde[k] @ x[:, :, k] + b.reshape(-1, 1)
        perturbed_b_all[:, k] = perturbed_b[:, 0]  # Ensuring proper shape
        b_hat, P, e_k_RLS = RLS_func(b_hat, P, f_factor, k, perturbed_b_all, order) 
        all_coeffs_list.append(np.append(b_hat[::-1], 1.0))

        ## Calculate error at the beginning and add it to current K error
        ## By adding the error of the RLS and the error at a given point best generalization results are seen!
        
        e_k_norm = window_error(b_hat, e_k_RLS, perturbed_b_all, k, order, 5)    
        norm_list.append(e_k_norm)
        
        ### If are norm improves we want to keep the computed coefficients
        if e_k_norm < e_k_norm_best:
            e_k_norm_best = e_k_norm
            b_hat_best = b_hat
            k_best = k

        if threshold_reached == True and e_k_norm > 1:
            e_k_norm_best = 1e3
            
        #### If there has been no improvements since 5 time steps ago, we assume a more or less stable solution
        ### We then trigger our control based algorithm to act on this system estimate. 
        ### Moreover this prevents us from having to recalculate often
        if k == k_best + 5:    
            threshold_reached = True
            test_coeffs = np.append(b_hat_best[::-1], 1.0) ## Note we have to reverse our b_hat and append a 1.0 for largest z order
            used_index.append(k-5)

        ## If we haven't reached our threshold use basic online gradient (code taken from Nicola)
        if threshold_reached == False:
            y_uncontrolled = x[...,k]
        
            y_uncontrolled = y_uncontrolled - step*f.gradient(y_uncontrolled, k*f.time.t_s)
            
            x[...,k+1] = y_uncontrolled

        ## Once we have reached a stable solution
        ## Run the original algorithm with our calculated coefficients, keeping track of how often we do it
        if threshold_reached == True:
            if np.any(test_coeffs != test_coeffs_prev):
                n = n + 1
                c = control_design(test_coeffs, lambda_lims)  
                test_coeffs_prev = test_coeffs
                
            e = - f.gradient(x[...,k], k*f.time.t_s)
                    
            y = y[1:] + [-sum([test_coeffs[i]*y[i]/test_coeffs[order] for i in range(order)]) + e/test_coeffs[order]]
            
            x[...,k+1] = sum([c[i]*y[i] for i in range(order)])

    ## Printing extra debugging info ##
    print(f"We have had to solve the LMIs {n} times.")
    print(f"The minimum error norm is {e_k_norm_best}")
    return x, test_coeffs, all_coeffs_list, used_index, norm_list

# Online gradient with compensation, designed using control theory
def online_gradient_control_sys_id_ARM(f, b, lambda_lims, delta, f_factor, step, A_tilde, x_0=0):
    
    ## Initialize needed values (More than just RLS due to added complexity)
    x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    order = 1
    dim = 0
    maxy = int(np.log(f.time.num_samples)**(1 + delta))
    y = [np.zeros(f.dom.shape) for _ in range(order)]
    b_hat = np.ones((order,1))
    b_hat_best = np.ones((order,1))
    e_k_norm_best = 1e3
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
    best_err_list = []
    offset = 0
    perturbed_b_all = np.zeros_like(b)
    Å = 0
    
    for k in range(f.time.num_samples):
        ## Here we grow the order based on log(k)^(1+delta)
        ## We must add some value error cases to let it run properly
        order_prev = order
        try:
            if k <= 3:
                raise ValueError("k must be greater than 0")
            
            growth = np.log(k - offset)**(1 + delta)
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

        ## Run RLS and append to history list. Also calculate our initial error
        ## By adding the error of the RLS and the error at a given point best generalization results are seen!
        perturbed_b = A_tilde[k] @ x[:, :, k] + b[:, k].reshape(-1, 1)
        #perturbed_b = A_tilde[k] @ x[:, :, k] + b.reshape(-1, 1)
        perturbed_b_all[:, k] = perturbed_b[:, 0]  # Ensuring proper shape
        b_hat, P, e_k_RLS = RLS_func(b_hat, P, f_factor, k, perturbed_b_all, order)  
        all_coeffs_list.append(np.append(b_hat[::-1], 1.0))
        e_k_norm = window_error(b_hat, e_k_RLS, perturbed_b_all, k, order, win_size=5)
        
        ## compute the new norm and add to history list
        norm_list.append(e_k_norm)
        ### If are norm improves we want to keep the computed coefficients
        ### We also calculate ahead to see if the order will grow within the next 5 timesteps
        if e_k_norm < e_k_norm_best and order != 1:
            e_k_norm_best = e_k_norm
            b_hat_best = b_hat
            k_best = k
            best_order = order
            order_k_best_5 = int(np.log(k_best - offset + 5) ** (1 + delta))
        
        if threshold_reached == True and window_error(b_hat_best, 0, perturbed_b_all, k, best_order, win_size=15) > window_error(b_hat_best, 0, perturbed_b_all, k-1, best_order, win_size=15) * 1e8:      
            print(f"Reset sys id at {k}")
            threshold_reached = False
            order = 1
            dim = 0
            y = [np.zeros(f.dom.shape) for _ in range(order)]
            b_hat = np.ones((order,1))
            e_k_norm_best = 1e-2
            P = np.eye(order) * 1
            c = np.ones(order+1)
            offset = k - 3
            
        ### if we have improved our error and we haven't had an improvement in the past fivetime steps, or 
        ### the order will grow within the next five timesteps, compute our estimated coefficients
        if k_best is not None and (k == k_best + 5 or best_order != order_k_best_5) and order > 1:
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
                
                y = [np.zeros(f.dom.shape) for _ in range(order)]
                y = reset_y(f, x, k, test_coeffs, c, order, y) 
                if order != dim:
                    #y = [np.zeros(f.dom.shape) for _ in range(order)]
                    Å += 1
                dim = order
            
            e = - f.gradient(x[...,k], k*f.time.t_s)
            
            y = y[1:] + [-sum([test_coeffs[i]*y[i]/test_coeffs[dim] for i in range(dim)]) + e/test_coeffs[dim]]
            
            x[...,k+1] = sum([c[i]*y[i] for i in range(dim)])
            
    ## Printing extra debugging info ##
    print(f"We have had to solve the LMIs {n} times.")
    print(f"Of which the order changed {Å} times.")
    print(f"The minimum error norm is {e_k_norm_best}")
    return x, test_coeffs, all_coeffs_list, used_index, norm_list


def reset_y(f, x, timestep, test_coeffs, c, order, y):
    copyx = x.copy()
    for k in range(timestep-500, timestep):
        e = - f.gradient(copyx[...,k], k*f.time.t_s)
            
        y = y[1:] + [-sum([test_coeffs[i]*y[i]/test_coeffs[order] for i in range(order)]) + e/test_coeffs[order]]
            
        copyx[...,k+1] = sum([c[i]*y[i] for i in range(order)])
    return y