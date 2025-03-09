#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la
import picos as pic

ran = np.random.default_rng()


from tvopt import costs, sets


#%% ONLINE OPTIMIZATION ALGORITHMS

# Online gradient for quadratic TV problems
def online_gradient(problem, step, x_0=0, num_iter=1):
    
    f = problem["f"]
    
    x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    
    
    for k in range(f.time.num_samples):
        
        y = x[...,k]
        
        for _ in range(num_iter):
        
            y = y - step*f.gradient(y, k*f.time.t_s)
        
        x[...,k+1] = y
    
    return x


# Online gradient with prediction for quadratic costs
def online_gradient_prediction(problem, step, x_0=0, num_iter=1):
    
    f = problem["f"]
    
    x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    
    
    for k in range(f.time.num_samples):
        
        y = x[...,k]
        
        for _ in range(num_iter):
        
            # compute prediction
            if k == 0: g = f.gradient(y, k*f.time.t_s)
            else: g = 2*f.gradient(y, k*f.time.t_s) - f.gradient(y, (k-1)*f.time.t_s)
            
            # gradient step on predicted cost
            y = y - step*g
        
        x[...,k+1] = y
        
    return x


#%% ONLINE OPTIMIZATION ALGORITHMS -- CONTROL-BASED DESIGN

# Online gradient with compensation, designed using control theory
def online_gradient_control(problem, c, x_0=0):
    
    f, b = problem["f"], problem["b"]
    if len(c) >= len(b): raise ValueError("The controller must be strictly proper (`c` of length strictly smaller than `b`).")
    
    # compute the coefficients for the output
    m = len(b)-1
    
    x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    y = [np.zeros(f.dom.shape) for _ in range(m)]
    
    
    for k in range(f.time.num_samples):
                
        e = - f.gradient(x[...,k], k*f.time.t_s)
                
        y = y[1:] + [-sum([b[i]*y[i]/b[m] for i in range(m)]) + e/b[m]]

        x[...,k+1] = sum([c[i]*y[i] for i in range(m)])
        
    return x


# Design a robust stabilizing controller that can be implemented in 
# `online_gradient_control`
def control_design(b, lambda_lims):

    # ------ open-loop system
    m = len(b)-1 # system size-1
    
    F = np.diag(np.ones(m-1), k=1)
    F[-1,:] = np.reshape([-b[i]/b[m] for i in range(m)], (1,-1))
    
    G = np.zeros((m, 1))
    G[-1] = -1/b[m]
    
    # ------ create problem
    prob = pic.Problem()
    
    # define variables
    P = pic.SymmetricVariable("P", m)
    W = pic.RealVariable("W", (1, m))
    
    # constraints
    prob.add_constraint(P >> 0)
    prob.add_constraint(((-P & F * P + lambda_lims[0] * G * W) // \
                         (P * F.T + lambda_lims[0] * W.T * G.T & -P)) << 0)
    prob.add_constraint(((-P & F * P + lambda_lims[1] * G * W) // \
                         (P * F.T + lambda_lims[1] * W.T * G.T & -P)) << 0)
    
    # solve
    try:
        prob.solve()
    except pic.SolutionFailure:
        print("The controlled could not be designed.")
        return None
    
    return la.solve(P, W.T).T


#%% COSTS

# Time-varying quadratic cost 0.5 x.T A_k x + x.T b_k
# in which both the quadratic and linear terms can change over time 
# (or only one of them)
class TVQuadratic(costs.Cost):
    
    def __init__(self, A, b, t_s=1):
        
        if isinstance(A, list) and isinstance(b, list):
            if len(A) != len(b):
                raise ValueError("`A` and `b` must have the same length.")
            n, num_samples = b[0].shape[0], len(b)
        elif isinstance(b, list): # A constant
            n, num_samples = b[0].shape[0], len(b)
            A = [A for _ in range(num_samples)]
        elif isinstance(A, list): # b constant
            n, num_samples = b.shape[0], len(A)
            b = [b for _ in range(num_samples)]
        else:
            raise ValueError("At least one of `A` and `b` must be a list.")
            
        super().__init__(sets.R(n, 1), sets.T(t_s, t_max=t_s*num_samples))
        self.A, self.b = A, b
        
        self.smooth = 2
    
    def function(self, x, t):
        
        k = round(t/self.time.t_s)
        return x.T.dot(0.5*self.A[k].dot(x) + self.b[k])
    
    def gradient(self, x, t):
        
        k = round(t/self.time.t_s)
        return self.A[k].dot(x) + self.b[k]

    def hessian(self, x, t):
        
        return self.A[round(t/self.time.t_s)]


class DynamicExample(costs.Cost):
    
    def __init__(self, t_s, t_max, n=1, omega=0.02*np.pi, phi=0, kappa=7.5, mu=1, omega_2=0.02*np.pi):
        
        super().__init__(sets.R(n, 1), sets.T(t_s, t_max=t_max))
        
        self.omega, self.kappa, self.omega_2 = omega, kappa, omega_2
        
        if np.isscalar(phi): self.phi = phi*np.ones(self.dom.shape)
        else: self.phi = np.reshape(phi, self.dom.shape)
        
        if np.isscalar(mu): self.mu = mu*np.ones(self.dom.shape)
        else: self.mu = np.reshape(mu, self.dom.shape)
            
        self.smooth = 2
    
    def function(self, x, t):
                
        return 0.5*la.norm(x - np.cos(self.omega*t + self.phi))**2 \
               + self.kappa*np.cos(self.omega_2*t)*np.log(1 + np.exp(self.mu.T.dot(x)))
    
    def gradient(self, x, t):
                
        return x - np.cos(self.omega*t + self.phi) \
               + self.kappa*np.cos(self.omega_2*t)*np.exp(self.mu.T.dot(x)) / (1 + np.exp(self.mu.T.dot(x))) * self.mu

    def hessian(self, x, t=None):
                
        return np.eye(self.dom.size) + self.kappa*np.cos(self.omega_2*t) \
               *np.exp(self.mu.T.dot(x)) / (1 + np.exp(self.mu.T.dot(x)))**2 * self.mu.dot(self.mu.T)

    def time_derivative(self, x, t, der="tx"):
        
        # parse the derivative order
        der = ''.join(der.split()).lower()
                
        # time der. gradient
        if der == "tx" or der == 'xt':
            return self.omega*np.sin(self.omega*t + self.phi) - self.kappa*self.omega_2*np.sin(self.omega_2*t) \
                   *np.exp(self.mu.T.dot(x)) / (1 + np.exp(self.mu.T.dot(x))) * self.mu
        else: return super().time_derivative(self.dom.check_input(x), t, der=der)


#%% UTILS

# Compute the coefficients of the denominator of the z-transform for a product
# of sines with frequencies given in `thetas`
def sine_product_z_transform(thetas):
    
    p = 1
    for t in thetas:
        p *= np.polynomial.Polynomial([1, -2*np.cos(t), 1])
    
    return p.coef


# Compute the product of polynomials from the coefficient list of each
def polynomial_product(*coeffs):
    
    p = 1
    
    for c in coeffs:
        p *= np.polynomial.Polynomial(c)
    
    return p.coef


# Closed-loop convergence rate
def convergence_rate(b, c, lambda_lims):
    
    B, C = np.polynomial.Polynomial(b), np.polynomial.Polynomial(c)
    
    p_min, p_max = B + lambda_lims[0] * C, B + lambda_lims[1] * C
    
    return np.abs((p_min.roots(), p_max.roots())).max()