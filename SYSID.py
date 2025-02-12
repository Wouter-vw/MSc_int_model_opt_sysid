import numpy as np
from tools import control_design
from tvopt import costs, sets


def RLS_func_reworked(b_hat, P, f_factor, k, b, order):
    # Form the regressor vector 
    h_k = np.array([b[:, k - i - 1] for i in range(order)])
    
    # Compute the error: e_k = b_k - h_k^T b_hat
    e_k_bef = b[:, k].reshape((len(b),1)) + h_k.T @ b_hat  

    ### Reset the P matrix to preserve stability
    if np.linalg.norm(P) > 1e16:
        P = np.eye(order) * 1 
    
    # Update RLS gain
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


# Online gradient with compensation, designed using control theory
def online_gradient_control_sys_id_rework(f, b, lambda_lims, order, f_factor,threshold, step, x_0=0):
    ## Initialize needed values
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

    for k in range(f.time.num_samples):
        ## Run sys ID and append the history of RLS ##
        b_hat, P, e_k_RLS = RLS_func_reworked(b_hat, P, f_factor, k, b, order)
        all_coeffs_list.append(np.append(b_hat[::-1], 1.0))

        ## Calculate error at the beginning and add it to current K error
        e_k_init = b[:, order+1].reshape((len(b),1)) + np.array([b[:, order - i] for i in range(order)]).T @ b_hat
        e_k_norm = np.linalg.norm(e_k_init, ord=1) + e_k_RLS
        norm_list.append(e_k_norm)
        ### If we reach the threshold and our norm improves, create the coefficients
        if e_k_norm < threshold and ~threshold_reached and e_k_norm < e_k_norm_best:
            threshold_reached = True
            e_k_norm_best = e_k_norm
            test_coeffs = np.append(b_hat[::-1], 1.0) ## Note we have to reverse our b_hat and append a 1.0 for largest z order
            used_index.append(k)

        ## If we haven't reached our threshold use online gradient
        if threshold_reached == False:
            y_uncontrolled = x[...,k]
        
            y_uncontrolled = y_uncontrolled - step*f.gradient(y_uncontrolled, k*f.time.t_s)
            
            x[...,k+1] = y_uncontrolled

        ## Run the original algorithm with our calculated coefficients, keeping track of how often we do it
        if threshold_reached == True:
            if np.any(test_coeffs != test_coeffs_prev):
                n = n + 1
                c = control_design(test_coeffs, lambda_lims)  
                test_coeffs_prev = test_coeffs
                
            e = - f.gradient(x[...,k], k*f.time.t_s)
                    
            y = y[1:] + [-sum([test_coeffs[i]*y[i]/test_coeffs[order] for i in range(order)]) + e/test_coeffs[order]]
            
            x[...,k+1] = sum([c[i]*y[i] for i in range(order)])

    print(f"We have had to solve the LMIs {n} times.")
    print(f"The minimum error norm is {e_k_norm_best}")
    return x, test_coeffs, all_coeffs_list, used_index, norm_list

# Online gradient with compensation, designed using control theory
def online_gradient_control_sys_id_ARX_rework(f, b, lambda_lims, delta, f_factor,threshold, normalized_signal, step, x_0=0):
    
    ## Initialize needed values
    x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    order = 1
    dim = 0
    maxy = int(np.log(f.time.num_samples)**(1 + delta))
    c = np.ones(order)
    y = [np.zeros(f.dom.shape) for _ in range(order)]
    b_hat = np.ones((order,1))
    e_k_norm_best = 1e3
    e_k_RLS = 1e3
    e_k_init = np.ones_like(b[:,0])*1e3
    P = np.eye(order) * 1  
    c = np.ones(order+1)
    n = 0
    test_coeffs_prev = np.zeros(1)
    test_coeffs = np.zeros(1)

    b_normed = np.empty_like(b)
    all_coeffs_list = []
    used_index = []
    threshold_reached = False
    norm_list = []
    for k in range(f.time.num_samples):
        ## Here we grow the order based on log(k)^(1+delta)
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
    ## Make sure to extend our b_hat and P
        if order != order_prev:
            b_hat = np.vstack((b_hat, [[1]]))
            P = np.pad(P, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            P[-1, -1] = 1

    ## If we want a normalized signal, for example if we have unbounded growth, set flag as true    
        if normalized_signal == True:     
            if k > 2:
                ## Run RLS on our normalized signal and append to history list. Also calculate our initial error
                b_normed = (b[:,:k]  - np.mean(b[:,:k], axis = 1, keepdims=True))/np.max(b[:,:k], axis = 1, keepdims=True)
                b_hat, P, e_k_RLS = RLS_func_reworked(b_hat, P, f_factor, k-2, b_normed, order)
                all_coeffs_list.append(np.append(b_hat[::-1], 1.0))
                e_k_init = b_normed[:, order+1].reshape((len(b),1)) + np.array([b_normed[:, order - i] for i in range(order)]).T @ b_hat
        else:
                ## Run RLS and append to history list. Also calculate our initial error
                b_hat, P, e_k_RLS = RLS_func_reworked(b_hat, P, f_factor, k, b, order)
                all_coeffs_list.append(np.append(b_hat[::-1], 1.0))
                e_k_init = b[:, order+1].reshape((len(b),1)) + np.array([b[:, order - i] for i in range(order)]).T @ b_hat
        
        e_k_norm = np.linalg.norm(e_k_init, ord=1) + e_k_RLS
        norm_list.append(e_k_norm)
    ### If we reach the threshold and our norm improves, create the coefficients
        if e_k_norm < threshold and ~threshold_reached and e_k_norm < e_k_norm_best:
            threshold_reached = True
            e_k_norm_best = e_k_norm
            test_coeffs = np.append(b_hat[::-1], 1.0)
            used_index.append(k)
        
    ## If we haven't reached our threshold use online gradient
        if threshold_reached == False:
            y_uncontrolled = x[...,k]
        
            y_uncontrolled = y_uncontrolled - step*f.gradient(y_uncontrolled, k*f.time.t_s)
            
            x[...,k+1] = y_uncontrolled
    
    ## Run the original algorithm with our calculated coefficients, keeping track of how often we do it
        if threshold_reached == True:
            if test_coeffs.shape != test_coeffs_prev.shape or np.any(test_coeffs != test_coeffs_prev):                
                n = n + 1
                c = control_design(test_coeffs, lambda_lims)
                test_coeffs_prev = test_coeffs
                if order != dim:
                    y = [np.zeros(f.dom.shape) for _ in range(order)]
                dim = order
            
            e = - f.gradient(x[...,k], k*f.time.t_s)
            
            y = y[1:] + [-sum([test_coeffs[i]*y[i]/test_coeffs[dim] for i in range(dim)]) + e/test_coeffs[dim]]
            
            x[...,k+1] = sum([c[i]*y[i] for i in range(dim)])
            
    
    print(f"We have had to solve the LMIs {n} times.")
    print(f"The minimum error norm is {e_k_norm_best}")

    return x, test_coeffs, all_coeffs_list, used_index, norm_list

## Function to calculate our delta error
def calculate_norm_error(coeffs_list, coeffs_sys):
    dif_list = []
    if type(coeffs_list[0]) == np.ndarray:
        for x in coeffs_list:
            if len(coeffs_sys) > len(x):
                dif = (len(coeffs_sys) - len(x))
                delta_error = np.linalg.norm(coeffs_sys - np.append(x, np.zeros(dif)), ord=np.inf)
                differs = True
            elif len(coeffs_sys) < len(x):
                dif = len(x) - len(coeffs_sys)                            
                delta_error = np.linalg.norm(x - np.append(coeffs_sys, np.zeros(dif)), ord=np.inf)
                differs = True
            else:
                delta_error = np.linalg.norm(coeffs_sys - x, ord=np.inf)
                differs = False
        
            dif_list.append(delta_error)
    else:
        if len(coeffs_sys) > len(coeffs_list):
            dif = (len(coeffs_sys) - len(coeffs_list))
            delta_error = np.linalg.norm(coeffs_sys - np.append(coeffs_list, np.zeros(dif)), ord=np.inf)
            differs = True
        elif len(coeffs_sys) < len(coeffs_list):
            dif = len(coeffs_list) - len(coeffs_sys)                            
            delta_error = np.linalg.norm(coeffs_list - np.append(coeffs_sys, np.zeros(dif)), ord=np.inf)
            differs = True
        else:
            delta_error = np.linalg.norm(coeffs_sys - coeffs_list, ord=np.inf)
            differs = False
        dif_list.append(delta_error)
    return dif_list, differs