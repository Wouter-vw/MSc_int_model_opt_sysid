import numpy as np
import matplotlib.pyplot as plt
import os


def plot_error_comparison_unknown_b_k(b_type, A_type, save_fig=True):
    # Construct the file path
    file_path = f"data/{b_type}_{A_type}/1-tv_linear_term-{b_type}_{A_type}.npz"
    
    # Load data
    data = np.load(file_path, allow_pickle=True)
    t_s = data["t_s"]
    t_max = data["t_max"]
    error_gradient = data["error_gradient"]
    error_control = data["error_control"]
    error_control_test = data["error_control_test"]
    
    if A_type == "constant":
        error_control_sys_id_ARM = data["error_control_sys_id_ARM"]

    t = np.arange(0,t_max,t_s)
    # Plot
    plt.figure()
    plt.semilogy(t, error_gradient, label="OGD")
    plt.semilogy(t, error_control, label="Control-based [1]", color='tab:orange')
    if A_type == "constant":
        plt.semilogy(t, error_control_sys_id_ARM, label="SIMBO-A", color = 'tab:red')
    plt.semilogy(t, error_control_test, label="SIMBO-B", color = 'tab:purple')

    plt.xlabel("Time")
    plt.ylabel("Tracking error")
    plt.legend()
    if A_type == "constant":
        plt.title(rf"Online Optimization with $\boldsymbol{{b}}_{{k}}$ = {b_type}")
    else:
        plt.title(rf"Online Optimization with $\boldsymbol{{b}}_{{k}}$ = {b_type} and {A_type} Hessian")

    # Save if requested
    if save_fig:
        save_path = f"data/{b_type}_{A_type}/1-error_comparison-{b_type}_{A_type}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    #plt.show()


def plot_error_comparison_known_b_k(b_type, save_fig):
    # Construct the file path
    file_path = f"Known_b_k/data/{b_type}/1-tv_linear_term-{b_type}.npz"
    
    # Load data
    data = np.load(file_path, allow_pickle=True)
    keys = ["t_s", "t_max", "error_gradient", "error_control",
        "error_control_sys_id", "error_control_sys_id_ARM"]
    t_s, t_max, error_gradient, error_control, error_control_sys_id, error_control_sys_id_ARM, = (data[k] for k in keys)

    t = np.arange(0,t_max,t_s)
    # Plot
    plt.figure()

    plt.semilogy(t, error_gradient, label="OGD")

    plt.semilogy(t, error_control, label="Control-based [1]", color = 'tab:orange')
    plt.semilogy(t[1:], error_control_sys_id[1:], label="SIMBO-A Adapted", color = 'tab:green')
    plt.semilogy(t[1:], error_control_sys_id_ARM[1:], label="SIMBO-A", color = 'tab:red')

    plt.xlabel("Time")
    plt.ylabel("Tracking error")
    plt.legend()
    plt.title(rf"Online Optimization with $\boldsymbol{{b}}_{{k}}$ = {b_type}")

    # Save if requested
    if save_fig:
        save_path = f"Known_b_k/data/{b_type}/1-tv_linear_term-{b_type}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    #plt.show()


## Function to calculate our delta error (Described in the paper for inexact model limit)
def calculate_norm_error(coeffs_list, coeffs_sys):
    ### Initialize empty list
    dif_list = []
    ## Fist is the case that our inputted coefficients are a list of arrays
    ## Deal with if the coefficient orders are different by appending zeros to whatever list is smaller,
    ## Then subtract the coefficients and compute the infinity norm
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
    ## Second case if it is just one coefficient array
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


### Plots the evolution of our delta error and the error we are using to obtain the best estimate for coefficients
### Also marks the indices where we compute controller coefficients. 
def plot_error_comparison_RLS(b_type, xrange, save_fig):
    
    # Construct the file path
    file_path = f"Known_b_k/data/{b_type}/1-tv_linear_term-{b_type}.npz"

    # Load data
    data = np.load(file_path, allow_pickle=True)
    keys = ["all_coeffs_list_RLS", "all_coeffs_list_ARM", "coeffs_sys", "used_index_RLS",
        "used_index_ARM", "norm_list_RLS", "norm_list_ARM"]
    all_coeffs_list_RLS, all_coeffs_list_ARM, coeffs_sys, used_index_RLS, used_index_ARM, norm_list_RLS, norm_list_ARM = (data[k] for k in keys)
    
    dif_list, _ = calculate_norm_error(all_coeffs_list_RLS, coeffs_sys)
    dif_list_arm, _ = calculate_norm_error(all_coeffs_list_ARM, coeffs_sys)

    used_rls = [dif_list[i] for i in used_index_RLS]
    used_arm = [dif_list_arm[i] for i in used_index_ARM]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    # Top subplot (RLS-related plots)
    ax1.semilogy(dif_list[:xrange], label='$||\delta||_1$ Error', color = 'tab:orange')
    ax1.semilogy(norm_list_RLS[:xrange], label='RLS Computed Error', color = 'tab:green')
    ax1.set_ylabel(r'$\||(\cdot)||_\infty$')
    ax1.set_title('SIMBO-A Adapted: True Error and Computed Error')
    ax1.scatter(used_index_RLS, used_rls, color='black', label='Controller Coeff. Computed$')
    ax1.grid(True)
    ax1.legend()

    # Bottom subplot (ARM-related plots)
    ax2.semilogy(dif_list_arm[:xrange], label='$||\delta||_1$ Error', color = 'tab:orange')
    ax2.semilogy(norm_list_ARM[:xrange], label='RLS Computed Error', color = 'tab:red')
    ax1.set_ylabel(r'$\||(\cdot)||_\infty$')
    ax2.set_xlabel('Timestep')
    ax2.set_title('SIMBO-A: True Error and Computed Error')
    ax2.scatter(used_index_ARM, used_arm, color='black', label='Controller Coeff. Computed')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if save_fig: plt.savefig(f"Known_b_k/data/{b_type}/1-norm_error_comp-{b_type}.pdf", bbox_inches="tight")
    plt.close()
    #plt.show()