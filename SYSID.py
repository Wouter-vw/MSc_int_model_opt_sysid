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

def window_error(b_hat, e_k_RLS, b, k, order, win_size):
    if k <= win_size:
        return 100  # No computation if k is within window size

    else: 
        win_err = e_k_RLS + sum(np.linalg.norm(
        b[:, k].reshape((len(b), 1)) + np.array([b[:, k - i - 1] for i in range(order)]).T @ b_hat,
        ord=1) for k in range(k - 1, k - win_size, -1))

    return win_err

# Online gradient with built in RLS
def online_gradient_control_sys_id_ARM(f, lambda_lims, e_k_best, delta, f_factor, step, x_0=1):
    x = np.ones(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    
    order = 1
    
    x_trial = np.zeros((f.dom.shape[0], f.time.num_samples))
    
    e_hat = np.ones((order,1))
    P = np.eye(order)
    
    e_hat_best = np.ones((order,1))
    threshold_reached = False
    test_coeffs_prev = 0
    
    for k in range(f.time.num_samples):
        if threshold_reached == False:
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
                e_hat = np.vstack((e_hat, [[1]]))
                P = np.pad(P, ((0, 1), (0, 1)), mode='constant', constant_values=0)
                P[-1, -1] = 1
            
            y = x[...,k]
            
            for _ in range(1):
            
                y = y - step*f.gradient(y, k*f.time.t_s)
            
            x[...,k+1] = y
        
            x_trial[...,k] = x[...,k+1].reshape(-1)
            
            e_hat, P, e_k_RLS = RLS_func(e_hat, P, f_factor, k, x_trial, order)
            e_k_norm = window_error(e_hat, e_k_RLS, x_trial, k-2, order, win_size=4)
            
            if e_k_norm < e_k_best:
                e_k_best = e_k_RLS
                e_hat_best = e_hat
                test_coeffs = np.append(e_hat_best[::-1], 1.0)
                threshold_reached = True

        if threshold_reached == True:
            if np.any(test_coeffs != test_coeffs_prev):
                c = control_design(test_coeffs, lambda_lims)  
                test_coeffs_prev = test_coeffs
                y = [np.zeros(f.dom.shape) for _ in range(order)]
                
            e = - f.gradient(x[...,k], k*f.time.t_s)
                    
            y = y[1:] + [-sum([test_coeffs[i]*y[i]/test_coeffs[order] for i in range(order)]) + e/test_coeffs[order]]
            
            x[...,k+1] = sum([c[i]*y[i] for i in range(order)])

    return x
