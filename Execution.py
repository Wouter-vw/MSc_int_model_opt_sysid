#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
ran = np.random.default_rng()

from tvopt import utils
from tools import TVQuadratic, online_gradient, online_gradient_control, control_design, sine_product_z_transform, polynomial_product
from SYSID import online_gradient_control_sys_id_ARM

#%% SET-UP
save_data = False

n = 15 # size of the unknown

t_s = 0.1 # sampling time
t_max = 200 # simulation length
num_samples = int(t_max/t_s) # total number of iterations


# quadratic cost matrix
L, mu = 5, 1 # smoothness and strong convexity parameters of the cost
if n > 1: A = utils.random_matrix(np.hstack((np.array([L]), (L-mu)*ran.random(n-2)+mu, np.array([mu]))))
else: A = (L-mu)*ran.random((1,1))+mu

random_values = np.random.rand(n)*0.1  # Generate random values between 0 and 1
decreasing_vector = np.sort(random_values)[::-1]

eigs, vectors = np.linalg.eigh(A)
A_tilde = [np.diag(vectors @ (np.sin(k*t_s)*decreasing_vector) @ vectors.T) for k in range(num_samples)]
A_list = [A + A_tilde[k] for k in range(num_samples)]

# step-size for online gradient
step = 2 / (L+mu)

# initial condition (random, used in all algorithms)
x0 = 50*ran.normal(size=(n,1))


#%% CHOOSE b_k

b_type = "constant" # "ramp" "sine" "sine+ramp" "sine^2"

if b_type == "constant":
    different_systems = False
    omega = 1
    b_bar = 5*ran.random((n,1)) # velocity
    b = b_bar*np.ones((n,num_samples))
    coeffs_sys = polynomial_product([-1, 1], [1, -2*np.cos(omega*t_s), 1], [1, -2*np.cos(2*omega*t_s), 1])


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

if b_type == "sine_inexact_B_D":
    different_systems = False
    omega = 1 # periodicity
    b = np.sin(omega*np.arange(0,t_max,t_s)*np.ones((n,1)))
    coeffs_sys = polynomial_product([-1, 1], [1, -2*np.cos(omega*t_s), 1], [1, -2*np.cos(2*omega*t_s), 1])


    
# compute optimal trajectory (used to compute the tracking error ||x_k - x_k^*||)
#x_opt = np.hstack([-la.solve(A,b[:,[k]]) for k in range(num_samples)])
x_opt = np.hstack([-la.solve(A_list[k], b[:,[k]]) for k in range(num_samples)])


# generate cost function
b_list = [b[:,[k]] for k in range(b.shape[1])]
f = TVQuadratic(A_list, b_list, t_s=t_s)

os.makedirs(f"data/{b_type}", exist_ok=True)


# ----- online gradient (Nicola's code)
print("Online Gradient Executing")
x = online_gradient({"f":f}, step, x_0=x0)
error_gradient = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]


# ------ control theoretical approach (Nicola's code)
print("Vanilla Approach Executing")
coeffs_ctrl = control_design(coeffs_sys, [mu, L])
x = online_gradient_control({"f":f, "b":coeffs_sys}, coeffs_ctrl, x_0=x0)
error_control = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

# ------ control theoretical with SYS ID
x = online_gradient_control_sys_id_ARM(f, [mu,L], e_k_best = 1e-6, delta = -0.13, f_factor = 0.1 , step = step, x_0=0)
error_control_ARX = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]

markevery = 1
fontsize = 16

t = np.arange(0,t_max,t_s)


plt.figure()
plt.semilogy(t, error_gradient, label="Online gradient")

plt.semilogy(t, error_control_ARX, label="Control-based Sys ID ARX")

plt.semilogy(t, error_control, label="Control-based")

# ------ asymptotic error ------
# computed as the maximum tracking error in the last 4/5ths of the simulation
tt = int(4*num_samples/5)

plt.xlabel("Time", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)
plt.legend(fontsize=fontsize-2)
plt.title(rf"Online Optimization with $\boldsymbol{{b}}_{{k}}$ = {b_type} and Changing Hessian")
plt.savefig(f"data/{b_type}/1-error_comparison-{b_type}.pdf", bbox_inches="tight")
plt.show()