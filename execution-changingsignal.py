#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
ran = np.random.default_rng()

from tvopt import utils
from tools import TVQuadratic, online_gradient, online_gradient_control, control_design, sine_product_z_transform, polynomial_product, convergence_rate
from SYSID import RLS_func, calculate_norm_error
from changing_signal import online_gradient_control_sys_id, online_gradient_control_sys_id_ARM, plot_error_comparison

### I have no yet tried all setting possible, if you try any code and encounter any bugs or errors/ issues please let me know!


#%% SET-UP
save_data = True

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

b_type = "ramp-then-sine" # "ramp" "sine" "sine+ramp" "sine^2"

if b_type == "ramp":
    different_systems = False
    b_bar = 5*ran.random((n,1)) # velocity
    b = np.arange(0,t_max,t_s)*b_bar + b_bar

    coeffs_sys = [1, -2, 1]
    coeffs_sys_2 = [0,0]

if b_type == "sine":
    different_systems = False
    omega = 2 # periodicity
    b = np.sin(omega*np.arange(0,t_max,t_s)*np.ones((n,1)))

    coeffs_sys = sine_product_z_transform(([omega*t_s]))
    coeffs_sys_2 = [0,0]
    
if b_type == "sine+ramp":
    different_systems = False
    omega = 1 # periodicity
    b = np.sin(omega*np.arange(0,t_max,t_s)*np.ones((n,1)))
    
    b_bar = 5*ran.random((n,1)) # velocity
    b += np.arange(0,t_max,t_s)*b_bar

    c_theta = np.cos(omega*t_s)
    coeffs_sys = sine_product_z_transform(([0, omega*t_s]))
    coeffs_sys_2 = [0,0]

if b_type == "sine^2":
    different_systems = False
    omega = 10 # periodicity
    b = np.sin(omega*np.arange(0,t_max,t_s)*np.ones((n,1)))**2

    coeffs_sys = polynomial_product([-1, 1], [1, -2*np.cos(omega*t_s), 1], [1, -2*np.cos(2*omega*t_s), 1])
    coeffs_sys_2 = [0,0]

if b_type == "sine-sine":
    different_systems = True
    omega1, omega2 = 1, 2
    t1 = np.arange(0, t_max / 2, t_s)
    t2 = np.arange(t_max / 2, t_max, t_s)
    
    b1 = np.sin(omega1 * t1) * np.ones((n, 1))  # First half with omega = 1
    b2 = np.sin(omega2 * t2) * np.ones((n, 1))  # Second half with omega = 2
    
    b = np.hstack((b1, b2))  # Concatenate along the time axis

    coeffs_sys = sine_product_z_transform(([omega1*t_s]))
    coeffs_sys_2 = sine_product_z_transform(([omega2*t_s]))

if b_type == "sine-sine^2":
    different_systems = True
    omega1, omega2 = 1, 10
    t1 = np.arange(0, t_max / 2, t_s)
    t2 = np.arange(t_max / 2, t_max, t_s)
    
    b1 = np.sin(omega1 * t1) * np.ones((n, 1))  # sine
    b2 = (np.sin(omega2 * t2) * np.ones((n,1)))**2  # sine^2
    
    b = np.hstack((b1, b2))  # Concatenate along the time axis

    coeffs_sys = sine_product_z_transform(([omega1*t_s]))
    coeffs_sys_2 = polynomial_product([-1, 1], [1, -2*np.cos(omega2*t_s), 1], [1, -2*np.cos(2*omega2*t_s), 1])

if b_type == "ramp-then-sine":
    different_systems = True
    omega1 = 2
    t1 = np.arange(0, t_max / 2, t_s)
    t2 = np.arange(t_max / 2, t_max, t_s)
    
    b1 = np.sin(omega1 * t2) * np.ones((n, 1))  # sine
    b_bar = 5*ran.random((n,1)) # velocity
    b2 = t1 *b_bar + b_bar

    b = np.hstack((b2, b1))  # Concatenate along the time axis
    
    coeffs_sys_2 = sine_product_z_transform(([omega1*t_s]))
    coeffs_sys = [1, -2, 1]

os.makedirs(f"data/changing_signal/{b_type}", exist_ok=True)

# compute optimal trajectory (used to compute the tracking error ||x_k - x_k^*||)
x_opt = np.hstack([-la.solve(A,b[:,[k]]) for k in range(num_samples)])

# generate cost function
b_list = [b[:,[k]] for k in range(b.shape[1])]
f = TVQuadratic(A, b_list, t_s=t_s)


#%% SIMULATIONS

# ----- online gradient (Nicola's code)
x = online_gradient({"f":f}, step, x_0=x0)
error_gradient = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

# ------ control theoretical approach (Nicola's code)
coeffs_ctrl = control_design(coeffs_sys, [mu, L])
x = online_gradient_control({"f":f, "b":coeffs_sys}, coeffs_ctrl, x_0=x0)
error_control = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]


#---------------- Recursive Least Squares implementation -----------------------
lambda_factor = 0.1  # Forgetting factor

true_order = len(coeffs_sys)-1  ### With the assumption that we have access to this

x, test_coeffs, all_coeffs_list_RLS, used_index_RLS, norm_list_RLS = online_gradient_control_sys_id(f, b, [mu, L], true_order, lambda_factor, step, different_systems, x_0=0)
delta_error_rls , differs_RLS = calculate_norm_error(test_coeffs, coeffs_sys)
error_control_sys_id = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

#---------------------  ARM(inf) implementation  --------------------------
lambda_factor = 0.1

### Just to show which delta which works well (This determines the growth rate of the order by log(n)^(1 + delta)###
if b_type == "sine+ramp":
    delta = 0.13
else: 
    delta = -0.13


x, test_coeffs, all_coeffs_list_ARM, used_index_ARM, norm_list_ARM = online_gradient_control_sys_id_ARM(f, b, [mu, L], delta, lambda_factor, step, x_0=0)
delta_error_ARM, differs_arm = calculate_norm_error(test_coeffs, coeffs_sys)
error_control_sys_id_ARM = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]
#-----------------------------------------------

#%% RESULTS

# ------ save data ------
if save_data:
    
    np.savez(f"data/changing_signal/{b_type}/1-tv_linear_term-{b_type}.npz", t_s=t_s, t_max=t_max, num_samples=num_samples,
             n=n, mu=mu, L=L, step=step, x0=x0, b_type=b_type,
             coeffs_sys=coeffs_sys, error_gradient=error_gradient, error_control=error_control, 
             error_control_sys_id = error_control_sys_id, delta_error_rls = delta_error_rls,
             error_control_sys_id_ARM = error_control_sys_id_ARM, delta_error_ARM = delta_error_ARM)


# ------ plot tracking error to visualize with info included ------
markevery = 1
fontsize = 16

t = np.arange(0,t_max,t_s)


plt.figure()

plt.semilogy(t, error_gradient, label="Online gradient")

plt.semilogy(t, error_control, label="Control-based")
plt.semilogy(t[1:], error_control_sys_id[1:], label="Control_with_sysID")
plt.semilogy(t[1:], error_control_sys_id_ARM[1:], label="Control_with_sysID_ARM")

# ------ asymptotic error ------
# computed as the maximum tracking error in the last 4/5ths of the simulation
tt = int(4*num_samples/5)

plt.xlabel("Time", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)
plt.legend(fontsize=fontsize-2)
plt.title(rf"Online Optimization with $\boldsymbol{{b}}_{{k}}$ = {b_type}")

if save_data: plt.savefig(f"data/changing_signal/{b_type}/1-error_comparison-{b_type}.pdf", bbox_inches="tight")
else: plt.show()

normalized_signal = False
plot_error_comparison(all_coeffs_list_RLS, all_coeffs_list_ARM, coeffs_sys, 
                          used_index_RLS, used_index_ARM, norm_list_RLS, norm_list_ARM, 2000, b_type, save_data, normalized_signal, coeffs_sys_2)