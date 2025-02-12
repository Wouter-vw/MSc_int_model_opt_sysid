#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import warnings
ran = np.random.default_rng()

from tvopt import utils
from tools import TVQuadratic, online_gradient, online_gradient_control, control_design, sine_product_z_transform, polynomial_product, convergence_rate
from SYSID import RLS_func_reworked, online_gradient_control_sys_id_rework, online_gradient_control_sys_id_ARX_rework, calculate_norm_error

save_data = False


#%% SET-UP

n = 15 # size of the unknown

t_s = 0.1 # sampling time
t_max = 200 # simulation length
num_samples = int(t_max/t_s) # total number of iterations


# quadratic cost matrix
L, mu = 5, 1 # smoothness and strong convexity parameters of the cost
if n > 1: A = utils.random_matrix(np.hstack((np.array([L]), (L-mu)*ran.random(n-2)+mu, np.array([mu]))))
else: A = (L-mu)*ran.random((1,1))+mu


# step-size for online gradient
step = 2 / (L+mu)

# initial condition (random, used in all algorithms)
x0 = 50*ran.normal(size=(n,1))


#%% CHOOSE b_k

b_type = "sine^2" # "ramp" "sine" "sine+ramp" "sine^2"

if b_type == "ramp":
    b_bar = 5*ran.random((n,1)) # velocity
    b = np.arange(0,t_max,t_s)*b_bar + b_bar

    coeffs_sys = [1, -2, 1]

if b_type == "sine":
    omega = 1 # periodicity
    b = np.sin(omega*np.arange(0,t_max,t_s)*np.ones((n,1)))

    coeffs_sys = sine_product_z_transform(([omega*t_s]))
    
if b_type == "sine+ramp":
    omega = 1 # periodicity
    b = np.sin(omega*np.arange(0,t_max,t_s)*np.ones((n,1)))
    
    b_bar = 5*ran.random((n,1)) # velocity
    b += np.arange(0,t_max,t_s)*b_bar

    c_theta = np.cos(omega*t_s)
    coeffs_sys = sine_product_z_transform(([0, omega*t_s]))

if b_type == "sine^2":
    omega = 10 # periodicity
    b = np.sin(omega*np.arange(0,t_max,t_s)*np.ones((n,1)))**2

    coeffs_sys = polynomial_product([-1, 1], [1, -2*np.cos(omega*t_s), 1], [1, -2*np.cos(2*omega*t_s), 1])

# compute optimal trajectory (used to compute the tracking error ||x_k - x_k^*||)
x_opt = np.hstack([-la.solve(A,b[:,[k]]) for k in range(num_samples)])

# generate cost function
b_list = [b[:,[k]] for k in range(b.shape[1])]
f = TVQuadratic(A, b_list, t_s=t_s)


#%% SIMULATIONS

# ----- online gradient
x = online_gradient({"f":f}, step, x_0=x0)

error_gradient = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]


# ------ control theoretical approach
# control design
start_time = time.time()
coeffs_ctrl = control_design(coeffs_sys, [mu, L])
end_time = time.time()
LMI_time = end_time - start_time
# apply algorithm

start_time = time.time()
x = online_gradient_control({"f":f, "b":coeffs_sys}, coeffs_ctrl, x_0=x0)
end_time = time.time()

vanilla_time = end_time - start_time
error_control = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]


## ---------------- RLS -----------------------
b_seq = b
lambda_factor = 0.1  # Forgetting factor
true_order = len(coeffs_sys)-1

threshold = 1e-10
start_time = time.time()
x, test_coeffs, all_coeffs_list_RLS, used_index_RLS, norm_list_RLS = online_gradient_control_sys_id_rework(f, b, [mu, L], true_order, lambda_factor, threshold, step, x_0=0)
end_time = time.time()
sysid_test_time = end_time - start_time

delta_error_rls , differs_RLS = calculate_norm_error(test_coeffs, coeffs_sys)
error_control_sys_id_tester = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]
#---------------------  ARX  --------------------------
threshold = 1e-8
delta = -0.13
lambda_factor = 0.1
normalized_signal = False
start_time = time.time()
x, test_coeffs, all_coeffs_list_ARM, used_index_ARM, norm_list_ARM = online_gradient_control_sys_id_ARX_rework(f, b, [mu, L], delta, lambda_factor, threshold, normalized_signal, step, x_0=0)
end_time = time.time()
sysid_test_time_ARX = end_time - start_time

delta_error_sysid , differs_arx = calculate_norm_error(test_coeffs, coeffs_sys)

error_control_sys_id_ARX = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]
#-----------------------------------------------



#%% RESULTS

# ------ save data
if save_data:
    
    np.savez(f"data/1-tv_linear_term-{b_type}.npz", t_s=t_s, t_max=t_max, num_samples=num_samples,
             n=n, mu=mu, L=L, step=step, x0=x0, b_type=b_type,
             coeffs_sys=coeffs_sys, error_gradient=error_gradient, error_control=error_control)



# ------ asymptotic error
# computed as the maximum tracking error in the last 4/5ths of the simulation
tt = int(4*num_samples/5)
# ------ plot tracking error
markevery = 1
fontsize = 16

t = np.arange(0,t_max,t_s)


plt.figure()

plt.semilogy(t, error_gradient, label="Online gradient")

plt.semilogy(t, error_control, label="Control-based")
plt.semilogy(t[1:], error_control_sys_id_tester[1:], label="Control_with_sysID")
plt.semilogy(t[1:], error_control_sys_id_ARX[1:], label="Control_with_sysID_ARX")


asymp_errors = f"Asymptotic Tracking Errors: \nOnline Gradient Method: {np.max(error_control[tt:]):.3e} \nRLS with known order: {np.max(error_control_sys_id_tester[tt:]):.3e} \nARM$(\infty)$: {np.max(error_control_sys_id_ARX[tt:]):.3e}" 
plt.figtext(1.0, 0.65, asymp_errors, fontsize=12, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8))


# Add text outside the plot area
if differs_arx == True:
    text = f"RLS Known order: $|| \delta_1 ||_{{\\infty}}$ = {delta_error_rls[0]:.3e} \n$ARM(\infty): || \delta_1 ||_{{\\infty}}$ = {delta_error_sysid[0]:.3e} \nNote: Internal Model and System ID model have different orders!"
else: 
    text = f"RLS Known order: $|| \delta_1 ||_{{\\infty}}$ = {delta_error_rls[0]:.3e} \n$ARM(\infty): || \delta_1 ||_{{\\infty}}$ = {delta_error_sysid[0]:.3e}"
plt.figtext(1.0, 0.45, text, fontsize=12, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8))


plt.xlabel("Time", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)
plt.legend(fontsize=fontsize-2)
plt.title(rf"Online Optimization with $\boldsymbol{{b}}_{{k}}$ = {b_type}")
#save_data = True
if save_data: plt.savefig(f"data/1-error_comparison-{b_type}.pdf", bbox_inches="tight")
else: plt.show()


#print(f"Vanilla LMI solve time: {LMI_time:.6f} seconds")
#print(f"Vanilla time: {vanilla_time:.6f} seconds")
#print(f"SysID test time: {sysid_test_time:.6f} seconds")
#print(f"SysID test time: {sysid_test_time_ARX:.6f} seconds")


dif_list , _ = calculate_norm_error(all_coeffs_list_RLS, coeffs_sys)
dif_list_arm, _ = calculate_norm_error(all_coeffs_list_ARM, coeffs_sys)

used_rls = [dif_list[i] for i in used_index_RLS]
used_arm = [dif_list_arm[i] for i in used_index_ARM]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

# Top subplot (RLS-related plots)
ax1.semilogy(dif_list[0:250], label='True Error RLS')
ax1.semilogy(norm_list_RLS[0:250], label='Computed Error Norm RLS')
ax1.set_ylabel(r'$\||(\cdot)||_\infty$')
ax1.set_title('RLS: True Error and Computed Error')
ax1.scatter(used_index_RLS, used_rls, color='red', label='Controller Computed ARM$(\infty)$')
ax1.grid(True)

# Bottom subplot (ARM-related plots)
ax2.semilogy(dif_list_arm[0:250], label='True Error ARM$(\infty)$')
if normalized_signal:
    ax2.semilogy(norm_list_ARM[3:253], label='Computed Error Norm ARM$(\infty)$')
else: 
    ax2.semilogy(norm_list_ARM[0:250], label='Computed Error Norm ARM$(\infty)$')

ax2.set_xlabel('Timestep')
ax2.set_title('ARM$(\infty)$: True Error and Computed Error')
ax2.scatter(used_index_ARM, used_arm, color='red', label='Controller Computed ARM$(\infty)$')
ax2.grid(True)

ax1.legend()
ax2.legend()

# Show the plot
plt.tight_layout()
plt.show()
