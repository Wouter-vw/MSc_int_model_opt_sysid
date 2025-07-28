import numpy as np
from numpy import linalg as la
from tvopt import utils
import matplotlib.pyplot as plt
from projected_funs import use_case_dissatisfaction, projected_online_gradient, projected_online_gradient_control_sys_id_ARM, anti_windup_projected_online_gradient_control, projection, projected_online_gradient_control, check_constraints
from SYSID import online_gradient_control_sys_id_ARM, control_design
from tools import online_gradient, online_gradient_control
import cvxpy as cp


ran = np.random.default_rng()
seed = np.random.seed(40)
save_data = True

DERs = 6 # size of the DERs
w_size = 2 # size of uncontrollables

t_s = 1 # sampling time
t_max = 8640 # simulation length
num_samples = int(t_max/t_s) # total number of iterations

# Desired eigenvalues
eigvals = np.array([1, 5])

# Random orthonormal matrix Q (6 x 2)
Q, _ = np.linalg.qr(np.random.randn(6, 2))

# Construct G: shape (2 x 6)
G = (np.sqrt(eigvals)[:, None] * Q.T)

beta = 1
A = G.T @ G 
w = np.sin(0.005*np.arange(0,t_max,t_s)*np.ones((w_size,1))) + np.ones((w_size,1))*2

L, mu = 5, 4
H = utils.random_matrix(np.hstack((np.array([L]), (L-mu)*ran.random(w_size-2)+mu, np.array([mu]))))

#### Hardcode y_ref ###
t1 = np.arange(0, t_max / 2, t_s)
t2 = np.arange(t_max / 2, t_max, t_s)
y1 = np.hstack((np.interp(t1, [t1[0], t1[-1]], [23, 39]).reshape(-1, 1), np.interp(t1, [t1[0], t1[-1]], [50, 30]).reshape(-1, 1)))
y2 = np.hstack((np.interp(t2, [t2[0], t2[-1]], [39, 26]).reshape(-1, 1), np.interp(t2, [t2[0], t2[-1]], [30, 33]).reshape(-1, 1)))
y_ref = np.vstack((y1, y2)).T

c = [np.linalg.norm(H @ w[:,k] - y_ref[:,k])**2 for k in range(w.shape[1])]
b_list = [(G.T @ (H @ w[:,k] - y_ref[:,k])).reshape((DERs,1)) for k in range(w.shape[1])]

# Compute list_h, first_entries, and second_entries
list_h = [H @ w[:, k] for k in range(w.shape[1])]
first_entries = [list_h[k][0] for k in range(len(list_h))]
second_entries = [list_h[k][1] for k in range(len(list_h))]

# Create figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# First subplot
axs[0].plot(y_ref[0], label=r"$y_{1,\mathrm{ref}}$")
axs[0].plot(first_entries, label=r"$h_{11}w_{1} + h_{12}w_{2}$")
axs[0].legend()
axs[0].set_title("PCC 1")
axs[0].set_ylim(bottom=0)  # y-axis starts at 0
axs[0].grid(True)


# Second subplot
axs[1].plot(y_ref[1], label=r"$y_{2,\mathrm{ref}}$")
axs[1].plot(second_entries, label=r"$h_{21}w_{1} + h_{22}w_{2}$")
axs[1].legend()
axs[1].set_title("PCC 2")
axs[1].set_ylim(bottom=0)  # y-axis starts at 0
axs[1].grid(True)

# Common labels
fig.supxlabel("Time (sec.)")
fig.supylabel("Power (kW)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for shared label
fig.savefig("Digital_Futures/data/dissatisfaction/pcc_comparison.pdf", format="pdf")

b_opt = np.stack(b_list)
b_opt = b_opt.reshape(8640, 6).T

#utils.random_matrix(np.hstack((np.array([L]), (L-mu)*ran.random(n-2)+mu, np.array([mu]))))
U1 = utils.random_matrix(np.hstack((np.array([1]), (1)*ran.random(6-2)+0, np.array([0]))))
U2 = np.random.randn(6, 1)
U3 = np.random.randn(1)

U4 = utils.random_matrix(np.hstack((np.array([1]), (1)*ran.random(6-2)+0, np.array([0]))))
U5 = np.random.randn(6, 1)
U6 = np.random.randn(1)

f = use_case_dissatisfaction(A, b_list, c, beta = 1, U1 = U1, U2 = U2, U3 = U3, U4 = U4, U5 = U5, U6 = U6,t_s=t_s)

# Store optimal x for each time step
x_opt_all = []

for k in range(num_samples):
    A_k = f.A[k]
    b_k = f.b[k]
    c_k = f.c[k]

    # Variables
    x = cp.Variable(DERs)
    z = cp.Variable(DERs, boolean=True)

    # Build disjoint constraints (same as before)
    constraints = []
    intervals = [
        ([-10, -6], [6, 10]),
        ([3, 7], [13, 17]),
        ([0, 3], [28, 32])
    ]
    for i in range(3):
        idx1 = 2 * i
        idx2 = idx1 + 1
        a1, b1 = intervals[i][0]
        a2, b2 = intervals[i][1]
        for j in [idx1, idx2]:
            constraints += [
                x[j] >= a1 * (1 - z[j]) + a2 * z[j],
                x[j] <= b1 * (1 - z[j]) + b2 * z[j]
            ]

    # Objective at time step k
    if k < 2160 or (4320 < k < 6480):
        u_k = 0.5 * cp.quad_form(x, U1) + U2.T @ x + U3
    else:
        u_k = 0.5 * cp.quad_form(x, U4) + U5.T @ x + U6
    
    objective = cp.Minimize(0.5 * cp.quad_form(x, A_k) + b_k.T @ x + 0.5 * c_k + u_k)

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS_BB)  # or use ECOS_BB if needed

    # Store solution
    x_opt_all.append(x.value)

# Stack solutions horizontally (each column = x_k)
x_opt = np.hstack(x_opt_all)
x_opt1 = x_opt.reshape(num_samples, 6).T.reshape(6, 1, num_samples)

step = 1/6
sets = [
    [(-10, -6), (6, 10)],    # For DER 0
    [(-10, -6), (6, 10)],    # For DER 1
    [(3, 7), (13, 17)],      # For DER 2
    [(3, 7), (13, 17)],      # For DER 3
    [(0, 3), (28, 32)],      # For DER 4
    [(0, 3), (28, 32)],      # For DER 5
]
x0 = np.array([[-6],[-6],[4],[4],[2],[2]])
print("Online Gradient Executing")
x = projected_online_gradient({"f":f}, step, sets, x_0=x0)
violations = check_constraints(x, sets)
error_gradient = [la.norm(x[...,k]-x_opt1[...,k]) for k in range(num_samples)]

coeffs_sys = [1, -2, 1]
# ------ control theoretical approach (Nicola's code)
print("Vanilla Approach Executing")
coeffs_ctrl = control_design(coeffs_sys, [0, 6])
x1 = projected_online_gradient_control({"f":f, "b":coeffs_sys}, coeffs_ctrl, sets, x_0=x0)
error_control = [la.norm(x1[...,k]-x_opt1[...,k]) for k in range(num_samples)]
violations = check_constraints(x1, sets)


print("Executing RLS with unknown order")
x2, test_coeffs= projected_online_gradient_control_sys_id_ARM(f, [0, 6], e_threshold = 1e-12, e_threshold2 = 1e-12, delta = 0.13, f_factor1 = 0.1, f_factor2 = 0.1, win_size1=4, win_size2=9, step=step, sets=sets, x_0=x0)
violations = check_constraints(x2, sets)
error_control_test = [la.norm(x2[...,k]-x_opt1[...,k]) for k in range(num_samples)]

# ----- online gradient (Nicola's code)
print("Online Gradient Executing")
x4 = online_gradient({"f":f}, step, x_0=x0)
for k in range(f.time.num_samples):
    x4[...,k+1] = projection(x4[...,k+1],sets)
error_gradient_nonprojected = [la.norm(x4[...,k]-x_opt1[...,k]) for k in range(num_samples)]
violations = check_constraints(x4, sets)

coeffs_sys = [1, -2, 1]
# ------ control theoretical approach (Nicola's code)
print("Vanilla Approach Executing")
coeffs_ctrl = control_design(coeffs_sys,[0, 6])
x5 = online_gradient_control({"f":f, "b":coeffs_sys}, coeffs_ctrl, x_0=x0)
for k in range(f.time.num_samples):
    x5[...,k+1] = projection(x5[...,k+1],sets)
error_control_nonprojected = [la.norm(x5[...,k]-x_opt1[...,k]) for k in range(num_samples)]
violations = check_constraints(x5, sets)


print("Executing RLS with unknown order")
x6, test_coeffs= online_gradient_control_sys_id_ARM(f, [0, 6], e_threshold = 1e-8, e_threshold2 = 1e-8, delta = 0.13, f_factor1 = 0.1, f_factor2 = 0.1, win_size1=4, win_size2=9, step=step, x_0=x0)
for k in range(f.time.num_samples):
    x6[...,k+1] = projection(x6[...,k+1],sets)
error_control_test_nonprojected = [la.norm(x6[...,k]-x_opt1[...,k]) for k in range(num_samples)]
violations = check_constraints(x6, sets)

coeffs_sys = [1, -2, 1]
# ------ control theoretical approach (Nicola's code)
print("Vanilla Approach Executing")
coeffs_ctrl = control_design(coeffs_sys, [0, 6])
x7 = anti_windup_projected_online_gradient_control({"f":f, "b":coeffs_sys}, coeffs_ctrl, sets, rho=0.4, x_0=x0)
error_control_antiwindup = [la.norm(x7[...,k]-x_opt1[...,k]) for k in range(num_samples)]
violations = check_constraints(x7, sets)

t = np.arange(0,t_max,t_s)
markevery = 1
fontsize = 16

plt.figure(figsize=(12, 6))
plt.semilogy(t, error_gradient, label="Projected Online gradient")
plt.semilogy(t, error_control_test, label="Projected SIMBO")
plt.semilogy(t, error_control, label="Projected CB [1]")
plt.semilogy(t, error_control_antiwindup, label="Projected CB [1] (Anti-windup)")
plt.semilogy(t, error_gradient_nonprojected, label="Online gradient")
plt.semilogy(t, error_control_test_nonprojected, label="SIMBO")
plt.semilogy(t, error_control_nonprojected, label="Control-based [1]")
plt.xlabel("Time", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)
plt.legend(fontsize=fontsize-2)
plt.title(rf"Online Optimization Compared to Constrained Optimum")
plt.tight_layout()
plt.savefig("Digital_Futures/data/dissatisfaction/tracking_error_plot.pdf")
plt.show()

t = np.arange(0,t_max,t_s)
markevery = 1
fontsize = 16
plt.figure(figsize=(12, 6))
plt.semilogy(t, np.cumsum(error_gradient), label="Projected Online gradient")
plt.semilogy(t, np.cumsum(error_control_test), label="Projected SIMBO")
plt.semilogy(t, np.cumsum(error_control), label="Projected CB [1]")
plt.semilogy(t, np.cumsum(error_control_antiwindup), label="Projected CB [1] (Anti-windup)")
plt.semilogy(t, np.cumsum(error_gradient_nonprojected), label="Online gradient")
plt.semilogy(t, np.cumsum(error_control_test_nonprojected), label="SIMBO")
plt.semilogy(t, np.cumsum(error_control_nonprojected), label="Control-based [1]")
plt.xlabel("Time", fontsize=fontsize)
plt.ylabel("Cumulative Tracking error", fontsize=fontsize)
plt.legend(fontsize=fontsize-2)
plt.title(rf"Online Optimization Compared to Constrained Optimum")
plt.tight_layout()
plt.savefig("Digital_Futures/data/dissatisfaction/cumulative_tracking_error_plot.pdf")
plt.show()

if save_data == True:
    np.savez("Digital_Futures/data/dissatisfaction/tracking_error_data.npz",
            t=t,
            t_max=t_max,
            t_s=t_s,
            error_gradient=error_gradient,
            error_control_test=error_control_test,
            error_control=error_control,
            error_control_antiwindup=error_control_antiwindup,
            error_gradient_nonprojected=error_gradient_nonprojected,
            error_control_test_nonprojected=error_control_test_nonprojected,
            error_control_nonprojected=error_control_nonprojected)

