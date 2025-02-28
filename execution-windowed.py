#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
ran = np.random.default_rng()

from tvopt import utils
from tools import TVQuadratic, online_gradient, online_gradient_control, control_design, sine_product_z_transform, polynomial_product, convergence_rate
from SYSID import RLS_func, online_gradient_control_sys_id, online_gradient_control_sys_id_ARM, calculate_norm_error, plot_error_comparison
from windowed_funcs import online_gradient_control_sys_id_ARM_window

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

b_type = "ramp" # "ramp" "sine" "sine+ramp" "sine^2"

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

os.makedirs(f"data/windowed/{b_type}", exist_ok=True)

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


#---------------------  ARM(inf) implementation  --------------------------
lambda_factor = 0.1

### Just to show which delta which works well (This determines the growth rate of the order by log(n)^(1 + delta)###
if b_type == "sine+ramp":
    delta = 0.13
else: 
    delta = -0.13

### We want to normalize unbounded signals such as ramp and sine+ramp as they continue to grow over time ###
if b_type == "sine+ramp":
    normalized_signal = True
else: 
    normalized_signal = False


x, test_coeffs, all_coeffs_list_ARM, used_index_ARM, norm_list_ARM = online_gradient_control_sys_id_ARM(f, b, [mu, L], delta, lambda_factor, normalized_signal, step, x_0=0)
error_control_sys_id_ARM = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

x, _, _, _, _ = online_gradient_control_sys_id_ARM_window(f, b, [mu, L], delta, lambda_factor, normalized_signal, step, win_size = 5, x_0=0)
error_control_sys_id_ARM_window5 = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

x, _, _, _, _ = online_gradient_control_sys_id_ARM_window(f, b, [mu, L], delta, lambda_factor, normalized_signal, step, win_size = 10, x_0=0)
error_control_sys_id_ARM_window10 = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

x, _, _, _, _ = online_gradient_control_sys_id_ARM_window(f, b, [mu, L], delta, lambda_factor, normalized_signal, step, win_size = 15, x_0=0)
error_control_sys_id_ARM_window15 = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

x, _, _, _, _ = online_gradient_control_sys_id_ARM_window(f, b, [mu, L], delta, lambda_factor, normalized_signal, step, win_size = 20, x_0=0)
error_control_sys_id_ARM_window20 = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

x, _, _, _, _ = online_gradient_control_sys_id_ARM_window(f, b, [mu, L], delta, lambda_factor, normalized_signal, step, win_size = 25, x_0=0)
error_control_sys_id_ARM_window25 = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

x, _, _, _, _ = online_gradient_control_sys_id_ARM_window(f, b, [mu, L], delta, lambda_factor, normalized_signal, step, win_size = 30, x_0=0)
error_control_sys_id_ARM_window30 = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]
#-----------------------------------------------

#%% RESULTS

# ------ save data ------
if save_data:
    
    np.savez(f"data/windowed/{b_type}/1-tv_linear_term-{b_type}.npz", t_s=t_s, t_max=t_max, num_samples=num_samples,
             n=n, mu=mu, L=L, step=step, x0=x0, b_type=b_type,
             coeffs_sys=coeffs_sys, error_gradient=error_gradient, error_control=error_control,
            error_control_sys_id_ARM = error_control_sys_id_ARM)


# ------ plot tracking error to visualize with info included ------
markevery = 1
fontsize = 16

t = np.arange(0,t_max,t_s)


plt.figure()

plt.semilogy(t, error_gradient, label="Online gradient")

plt.semilogy(t, error_control, label="Control-based (Original)")
plt.semilogy(t[1:], error_control_sys_id_ARM[1:], label="ARM No Window")
plt.semilogy(t[1:], error_control_sys_id_ARM_window5[1:], label="ARM Window Size 5")
plt.semilogy(t[1:], error_control_sys_id_ARM_window10[1:], label="ARM Window Size 10")
plt.semilogy(t[1:], error_control_sys_id_ARM_window15[1:], label="ARM Window Size 15")
plt.semilogy(t[1:], error_control_sys_id_ARM_window20[1:], label="ARM Window Size 20")
plt.semilogy(t[1:], error_control_sys_id_ARM_window25[1:], label="ARM Window Size 25")
plt.semilogy(t[1:], error_control_sys_id_ARM_window30[1:], label="ARM Window Size 30")
# ------ asymptotic error ------
# computed as the maximum tracking error in the last 4/5ths of the simulation
tt = int(4*num_samples/5)
# computed as the maximum tracking error in the last 4/5ths of the simulation

plt.xlabel("Time", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.title(rf"Online Optimization with $\boldsymbol{{b}}_{{k}}$ = {b_type} for Different Window Sizes")

if save_data: plt.savefig(f"data/windowed/{b_type}/1-error_comparison-{b_type}.pdf", bbox_inches="tight")
else: plt.show()