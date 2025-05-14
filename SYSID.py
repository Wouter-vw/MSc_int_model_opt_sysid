import numpy as np
from tools import control_design
from tvopt import costs, sets
import matplotlib.pyplot as plt

def RLS_func(b_hat, P, f_factor, k, b, order):
    # Form the regressor vector with the order which is given at k given
    h_k = np.array([b[:, k - i - 1] for i in range(order)])
    
    # Compute the error: e_k = b_k - h_k^T b_hat
    e_k_bef = b[:, k].reshape((len(b),1)) + h_k.T @ b_hat  

    ### Reset the P matrix to preserve stability (This must be done sometimes as the signal may not excite RLS enough!
    if np.linalg.norm(P) > 1e12:
        P = np.eye(order) * 1 
    
    # Update RLS gain (In the case that we encounter a singular matrix we try a potential more stable calculation, in practice has not be activated. 
    try:
        K_k = P @ h_k @ np.linalg.inv(np.eye(len(b)) * f_factor + h_k.T @ P @ h_k)
    except np.linalg.LinAlgError:
        print("Error: Singular matrix encountered.")
        K_k = P @ h_k @ np.linalg.pinv(np.eye(len(b))* f_factor + h_k.T @ P @ h_k)

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
def online_gradient_control_sys_id_ARM(f, lambda_lims, e_threshold, e_threshold2, delta, f_factor1, f_factor2, win_size1, win_size2, step, x_0=1):
    x = np.ones(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    
    order = 1
    
    x_trial = np.zeros((f.dom.shape[0], f.time.num_samples))
    
    e_hat = np.ones((order,1))
    P = np.eye(order)
    
    e_hat_best = np.ones((order,1))
    threshold_reached = False
    test_coeffs_prev = [0]
    kbest = f.time.num_samples
    offset = 0

    test_coeffs = [1, 1]

    dim_previous = -1

    for k in range(f.time.num_samples):
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
        if order != 1 and order != order_prev:
            e_hat = np.vstack((e_hat, [[1]]))
            P = np.pad(P, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            P[-1, -1] = 1
        
        if threshold_reached == True:
            if len(test_coeffs) != len(test_coeffs_prev) or np.any(test_coeffs != test_coeffs_prev):
                krecomp = k 
                print(f"Recompute at {k}")
                try:
                    c = control_design(test_coeffs, lambda_lims)  
                except:
                    print("No solution found")
                    c = control_design(test_coeffs_prev, lambda_lims) 
                    test_coeffs = test_coeffs_prev
                    
                test_coeffs_prev = test_coeffs
                dim = len(test_coeffs)-1

                if dim != dim_previous:
                    y = [np.zeros(f.dom.shape) for _ in range(dim)]
                    dim_previous = dim

            e = - f.gradient(x[...,k], k*f.time.t_s)
      
            y = y[1:] + [-sum([test_coeffs[i]*y[i]/test_coeffs[dim] for i in range(dim)]) + e/test_coeffs[dim]]
            
            x[...,k+1] = sum([c[i]*y[i] for i in range(dim)])

            x_trial[...,k] = x[...,k+1].reshape(-1)

            e_hat, P, e_k_RLS = RLS_func(e_hat, P, f_factor2, k, x_trial, order)
            e_k_norm = window_error(e_hat, e_k_RLS, x_trial, k-1, order, win_size=win_size2)

            if e_k_norm < e_k_best:
                e_k_best = e_k_norm
                e_hat_best = e_hat
                kbest = k

            if kbest + 20 == k:
                test_coeffs = np.append(e_hat_best[::-1], 1.0)

            if window_error(e_hat_best, 0, x_trial, k, len(e_hat_best), win_size=2) > window_error(e_hat_best, 0, x_trial, k-1, len(e_hat_best), win_size=4)*1e2 and k > krecomp + 20: 
                threshold_reached = False
                print(f"Reset sys id at {k}")
                order = 1
                order_prev = 1
                dim = 0
                dim_previous = -1
                y = [np.zeros(f.dom.shape) for _ in range(order)]
                e_hat = np.ones((order,1))
                e_threshold = e_threshold2
                P = np.eye(order) * 1
                c = np.ones(order+1)
                offset = k - 3
            
        if threshold_reached == False and k != offset:
            y_unc = x[...,k]
            
            for _ in range(1):
            
                y_unc = y_unc - step*f.gradient(y_unc, k*f.time.t_s)
            
            x[...,k+1] = y_unc
        
            x_trial[...,k] = x[...,k+1].reshape(-1)
            
            e_hat, P, e_k_RLS = RLS_func(e_hat, P, f_factor1, k, x_trial, order)
            e_k_norm = window_error(e_hat, e_k_RLS, x_trial, k-1, order, win_size=win_size1)
            
            if e_k_norm < e_threshold:
                e_hat_best = e_hat
                test_coeffs = np.append(e_hat_best[::-1], 1.0)
                threshold_reached = True
                e_k_best = e_threshold
                
    return x, test_coeffs