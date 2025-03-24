import numpy as np
from tools import control_design
from tvopt import costs, sets
import matplotlib.pyplot as plt
from SYSID import RLS_func, calculate_norm_error
from windowed_funcs import window_error

# Online gradient with built in RLS
def online_gradient_control_sys_id(f, b, lambda_lims, order, f_factor, step, different_systems, x_0=0):
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

    ### This is the main loop for our system ID
    for k in range(f.time.num_samples):
        ## Run sys ID and append the history of RLS ##
        b_hat, P, e_k_RLS = RLS_func(b_hat, P, f_factor, k, b, order)
        all_coeffs_list.append(np.append(b_hat[::-1], 1.0))

        ## Calculate error at the beginning and add it to current K error
        ## By adding the error of the RLS and the error at a given point best generalization results are seen!
        e_k_norm = window_error(b_hat, e_k_RLS, b, k, order, 15)    
        norm_list.append(e_k_norm)
        
        ### If are norm improves we want to keep the computed coefficients
        if e_k_norm < e_k_norm_best:
            e_k_norm_best = e_k_norm
            b_hat_best = b_hat
            k_best = k

        if threshold_reached == True and e_k_norm > 1e-1:
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
def online_gradient_control_sys_id_ARM(f, b, lambda_lims, delta, f_factor, step, x_0=0):
    
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
        b_hat, P, e_k_RLS = RLS_func(b_hat, P, f_factor, k, b, order)
        all_coeffs_list.append(np.append(b_hat[::-1], 1.0))
        e_k_norm = window_error(b_hat, e_k_RLS, b, k, order, win_size=15)
        
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
        
        if threshold_reached == True and window_error(b_hat_best, 0, b, k, best_order, win_size=15) > window_error(b_hat_best, 0, b, k-1, best_order, win_size=15) * 1e8:      
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

### Plots the evolution of our delta error and the error we are using to obtain the best estimate for coefficients
### Also marks the indices where we compute controller coefficients. 
def plot_error_comparison(all_coeffs_list_RLS, all_coeffs_list_ARM, coeffs_sys, 
                          used_index_RLS, used_index_ARM, norm_list_RLS, norm_list_ARM, xrange, b_type, save_data, normalized_signal, secondsys = [0,0]):

    if secondsys[0] != 0:
        dif_list, _ = calculate_norm_error(all_coeffs_list_RLS[0:999], coeffs_sys)
        list2, _ = calculate_norm_error(all_coeffs_list_RLS[1000:1999], secondsys)
        dif_list.extend(list2)
    else:
        dif_list, _ = calculate_norm_error(all_coeffs_list_RLS, coeffs_sys)

    if secondsys[0] != 0:
        dif_list_arm, _ = calculate_norm_error(all_coeffs_list_ARM[0:999], coeffs_sys)
        list2_arm, _ = calculate_norm_error(all_coeffs_list_ARM[1000:1999], secondsys)
        dif_list_arm.extend(list2_arm)
    else:
        dif_list_arm, _ = calculate_norm_error(all_coeffs_list_ARM, coeffs_sys)

    used_rls = [dif_list[i] for i in used_index_RLS]
    
    if normalized_signal == True:
        used_arm = [dif_list_arm[i] for i in [i - 3 for i in used_index_ARM]]
    else:
        used_arm = [dif_list_arm[i] for i in used_index_ARM]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    # Top subplot (RLS-related plots)
    ax1.semilogy(dif_list[:xrange], label='True Error RLS', zorder=1)
    ax1.semilogy(norm_list_RLS[:xrange], label='Computed Error Norm RLS', zorder=2)
    ax1.set_ylabel(r'$\||(\cdot)||_\infty$')
    ax1.set_title('RLS: True Error and Computed Error')
    ax1.scatter(used_index_RLS, used_rls, color='red', label='Controller Computed ARM$(\infty)$', zorder=3)
    ax1.grid(True)
    ax1.legend()

    # Bottom subplot (ARM-related plots)
    ax2.semilogy(dif_list_arm[:xrange], label='True Error ARM$(\infty)$')
    if normalized_signal == True:
        ax2.semilogy(norm_list_ARM[3:xrange+3], label='Computed Error Norm ARM$(\infty)$')
    else:
        ax2.semilogy(norm_list_ARM[:xrange], label='Computed Error Norm ARM$(\infty)$')
    ax2.set_xlabel('Timestep')
    ax2.set_title('ARM$(\infty)$: True Error and Computed Error')
    if normalized_signal == True:
        ax2.scatter([i - 3 for i in used_index_ARM], used_arm, color='red', label='Controller Computed ARM$(\infty)$')
    else:
        ax2.scatter(used_index_ARM, used_arm, color='red', label='Controller Computed ARM$(\infty)$')
    
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    if save_data: plt.savefig(f"data/changing_signal/{b_type}/1-norm_error_comp-{b_type}.pdf", bbox_inches="tight")
    else: plt.show()

def window_error(b_hat, e_k_RLS, b, k, order, win_size):
    if k <= win_size:
        return 100  # No computation if k is within window size

    else: 
        win_err = e_k_RLS + sum(np.linalg.norm(
        b[:, k].reshape((len(b), 1)) + np.array([b[:, k - i - 1] for i in range(order)]).T @ b_hat,
        ord=1) for k in range(k - 1, k - win_size, -1))

    return win_err