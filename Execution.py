#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import warnings
ran = np.random.default_rng()

from tvopt import utils
from tools import TVQuadratic, online_gradient, online_gradient_control, control_design, sine_product_z_transform, polynomial_product, convergence_rate
from SYSID import online_gradient_control_sys_id_ARM

#%% SET-UP
save_data = False

n = 15 # size of the unknown

t_s = 0.1 # sampling time
t_max = 800 # simulation length
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

b_type = "sine^2-ramp-mixed" # "ramp" "sine" "sine+ramp" "sine^2" "sine-sine" "sine^2-sine" "ramp-then-sine" "constant"
A_type = "constant" # "constant" "time-varying"

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

if b_type == "sine^2-sine":
    different_systems = True
    omega1, omega2 = 1, 10
    t1 = np.arange(0, t_max / 2, t_s)
    t2 = np.arange(t_max / 2, t_max, t_s)
    
    b1 = np.sin(omega1 * t1) * np.ones((n, 1))  # sine
    b2 = (np.sin(omega2 * t2) * np.ones((n,1)))**2  # sine^2
    
    b = np.hstack((b2, b1))  # Concatenate along the time axis

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

if b_type == "sine-sine^2-mixed":
    different_systems = False
    omega1, omega2 = 1, 10
    
    n1 = (n + 1) // 2
    n2 = n // 2
    
    b1 = np.sin(omega1*np.arange(0,t_max,t_s)*np.ones((n1,1)))
    b2 = np.sin(omega2*np.arange(0,t_max,t_s)*np.ones((n2,1)))**2
    b = np.vstack((b1,b2))
    coeffs_sys = polynomial_product([-1, 1], [1, -2*np.cos(omega2*t_s), 1], [1, -2*np.cos(2*omega2*t_s), 1])

if b_type == "sine-ramp-mixed":
    different_systems = False
    omega = 1
    n1 = (n + 1) // 2
    n2 = n // 2
    
    b1 = np.sin(omega*np.arange(0,t_max,t_s)*np.ones((n1,1)))
    b_bar = 5*ran.random((n2,1)) # velocity
    b2 = np.arange(0,t_max,t_s)*b_bar + b_bar
    b = np.vstack((b1,b2))
    coeffs_sys = sine_product_z_transform(([omega*t_s]))

if b_type == "sine-sine+ramp-mixed":
    different_systems = False
    omega1, omega2 = 1, 1
    n1 = (n + 1) // 2
    n2 = n // 2
    
    b1 = np.sin(omega1*np.arange(0,t_max,t_s)*np.ones((n1,1)))
    b2 = np.sin(omega2*np.arange(0,t_max,t_s)*np.ones((n2,1)))
    b_bar = 5*ran.random((n2,1)) # velocity
    b2 += np.arange(0,t_max,t_s)*b_bar
    b = np.vstack((b1,b2))
    coeffs_sys = sine_product_z_transform(([0, omega*t_s]))

if b_type == "sine^2-ramp-mixed":
    different_systems = False
    omega = 10
    n1 = (n + 1) // 2
    n2 = n // 2
    b1 = np.sin(omega*np.arange(0,t_max,t_s)*np.ones((n1,1)))**2
    b_bar = 5*ran.random((n2,1)) # velocity
    b2 = np.arange(0,t_max,t_s)*b_bar + b_bar
    b = np.vstack((b1,b2))
    coeffs_sys = polynomial_product([-1, 1], [1, -2*np.cos(omega*t_s), 1], [1, -2*np.cos(2*omega*t_s), 1])

if b_type == "sine^2-sine+ramp-mixed":
    different_systems = False
    omega1, omega2 = 10, 1
    n1 = (n + 1) // 2
    n2 = n // 2
    b1 = np.sin(omega1*np.arange(0,t_max,t_s)*np.ones((n1,1)))**2
    b2 = np.sin(omega2*np.arange(0,t_max,t_s)*np.ones((n2,1)))
    b_bar = 5*ran.random((n2,1)) # velocity
    b2 += np.arange(0,t_max,t_s)*b_bar
    b = np.vstack((b1,b2))
    coeffs_sys = polynomial_product([-1, 1], [1, -2*np.cos(omega1*t_s), 1], [1, -2*np.cos(2*omega1*t_s), 1])

if b_type == "ramp-sine+ramp-mixed":
    different_systems = False
    omega1 = 1
    n1 = (n + 1) // 2
    n2 = n // 2
    b_bar = 5*ran.random((n1,1)) # velocity
    b1 = np.arange(0,t_max,t_s)*b_bar + b_bar
    b2 = np.sin(omega1*np.arange(0,t_max,t_s)*np.ones((n2,1)))
    b_bar = 5*ran.random((n2,1)) # velocity
    b2 += np.arange(0,t_max,t_s)*b_bar
    b = np.vstack((b1,b2))
    coeffs_sys = sine_product_z_transform(([0, omega*t_s]))

# generate cost function
b_list = [b[:,[k]] for k in range(b.shape[1])]

if A_type == "constant":    
    x_opt = np.hstack([-la.solve(A,b[:,[k]]) for k in range(num_samples)])
    f = TVQuadratic(A, b_list, t_s=t_s)
if A_type == "time-varying":
    f = TVQuadratic(A_list, b_list, t_s=t_s)
    x_opt = np.hstack([-la.solve(A_list[k], b[:,[k]]) for k in range(num_samples)])

os.makedirs(f"data/{b_type}_{A_type}", exist_ok=True)


# ----- online gradient (Nicola's code)
print("Online Gradient Executing")
x = online_gradient({"f":f}, step, x_0=x0)
error_gradient = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]


# ------ control theoretical approach (Nicola's code)

print("Vanilla Approach Executing")
coeffs_ctrl = control_design(coeffs_sys, [mu, L])
x = online_gradient_control({"f":f, "b":coeffs_sys}, coeffs_ctrl, x_0=x0)
error_control = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]


x, test_coeffs= online_gradient_control_sys_id_ARM(f, [mu,L], e_threshold = 1e-12, e_threshold2 = 1e-12, delta = -0.13, f_factor1 = 0.1, f_factor2 = 0.1, win_size1=4, win_size2=9, step=step, x_0=0)
error_control_test = [la.norm(x[...,k]-x_opt[:,[k]]) for k in range(num_samples)]


markevery = 1
fontsize = 16

t = np.arange(0,t_max,t_s)


plt.figure()
plt.semilogy(t, error_gradient, label="Online gradient")

plt.semilogy(t, error_control_test, label="Control-based Sys ID ARX")

plt.semilogy(t, error_control, label="Control-based Baseline")

# ------ asymptotic error ------
# computed as the maximum tracking error in the last 4/5ths of the simulation
tt = int(4*num_samples/5)

plt.xlabel("Time", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)
plt.legend(fontsize=fontsize-2)
plt.title(rf"Online Optimization with $\boldsymbol{{b}}_{{k}}$ = {b_type} and {A_type} Hessian")
plt.savefig(f"data/{b_type}_{A_type}/1-error_comparison-{b_type}_{A_type}.pdf", bbox_inches="tight")
plt.show()
