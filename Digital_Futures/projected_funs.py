import numpy as np
ran = np.random.default_rng()
from tvopt import costs, sets
from tools import control_design
from SYSID import RLS_func, window_error

class use_case(costs.Cost):
    
    def __init__(self, A, b, c, beta, t_s=1):
        
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
        
        if isinstance(b, list) and isinstance(c, list):
            if len(b) != len(c):
                raise ValueError("`c` and `b` must have the same length.")
        
        super().__init__(sets.R(n, 1), sets.T(t_s, t_max=t_s*num_samples))
        self.A, self.b, self.c, self.beta = A, b, c, beta
        
        self.smooth = 2
    
    def function(self, x, t):
        
        k = round(t/self.time.t_s)
        return x.T.dot(0.5*self.beta*self.A[k].dot(x) + self.beta * self.b[k]) + 0.5 * self.beta * self.c[k]
        
    def gradient(self, x, t):
        
        k = round(t/self.time.t_s)
        return self.beta * self.A[k].dot(x) + self.beta * self.b[k]


class use_case_dissatisfaction(costs.Cost):
    
    def __init__(self, A, b, c, beta, U1 , U2, U3, U4, U5, U6, t_s=1):
        
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
        
        if isinstance(b, list) and isinstance(c, list):
            if len(b) != len(c):
                raise ValueError("`c` and `b` must have the same length.")
        
        super().__init__(sets.R(n, 1), sets.T(t_s, t_max=t_s*num_samples))
        self.A, self.b, self.c, self.beta = A, b, c, beta
        self.U1, self.U2, self.U3, self.U4, self.U5, self.U6 = U1, U2, U3, U4, U5, U6
        self.smooth = 2
    
    def function(self, x, t):
        
        k = round(t/self.time.t_s)
        base = x.T.dot(0.5*self.beta*self.A[k].dot(x) + self.beta * self.b[k]) + 0.5 * self.beta * self.c[k]

        if t < 2160 or (t > 4320 and t < 6480):
            u = x.T.dot(0.5*self.U1.dot(x) + self.U2) + self.U3
        else:
            u = x.T.dot(0.5*self.U4.dot(x) + self.U5) + self.U6
        return base + u
        
    def gradient(self, x, t):
        
        k = round(t/self.time.t_s)
        grad = self.beta * self.A[k].dot(x) + self.beta * self.b[k]
        
        if t < 2160 or (t > 4320 and t < 6480):
            grad += self.U1.dot(x) + self.U2
        else:
            grad += self.U4.dot(x) + self.U5
        return grad

def projection(x, sets):
    x_proj = np.zeros_like(x)

    for i in range(6):
        intervals = sets[i]
        a1, b1 = intervals[0]
        a2, b2 = intervals[1]
        xi = x[i]

        if a1 <= xi <= b1 or a2 <= xi <= b2:
            x_proj[i] = xi
        elif xi < a1:
            x_proj[i] = a1
        elif b1 < xi < a2:
            # Choose closer boundary
            if abs(xi - b1) <= abs(xi - a2):
                x_proj[i] = b1
            else:
                x_proj[i] = a2
        elif xi > b2:
            x_proj[i] = b2
        elif a1 <= xi < a2:
            # Between sets, choose boundary depending on sign (if needed)
            if xi <= (a1 + b1) / 2:
                x_proj[i] = b1
            else:
                x_proj[i] = a2
        else:
            # Shouldn't hit this case
            x_proj[i] = xi

    return x_proj


# Online gradient for quadratic TV problems
def projected_online_gradient(problem, step, sets, x_0=0, num_iter=1):
    
    f = problem["f"]
    
    x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    
    
    for k in range(f.time.num_samples):
        
        y = x[...,k]
        
        for _ in range(num_iter):
        
            y = y - step*f.gradient(y, k*f.time.t_s)
            y = projection(y,sets)
        
        x[...,k+1] = y
    
    return x

# Online gradient with built in RLS
def projected_online_gradient_control_sys_id_ARM(f, lambda_lims, e_threshold, e_threshold2, delta, f_factor1, f_factor2, win_size1, win_size2, step, sets, x_0=1):
    x = np.ones(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    
    order = 1
    
    x_trial = np.zeros((f.dom.shape[0], f.time.num_samples))
    
    e_hat = np.ones((order,1))
    P = np.eye(order)
    
    e_hat_best = np.ones((order,1))
    threshold_reached = False
    test_coeffs_prev = [0]
    kbest = f.time.num_samples
    offset = 0

    test_coeffs = [1, 1]

    dim_previous = -1

    for k in range(f.time.num_samples):
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
        if order != 1 and order != order_prev:
            e_hat = np.vstack((e_hat, [[1]]))
            P = np.pad(P, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            P[-1, -1] = 1
        
        if threshold_reached == True:
            if len(test_coeffs) != len(test_coeffs_prev) or np.any(test_coeffs != test_coeffs_prev):
                krecomp = k 
                print(f"Recompute at {k}")
                try:
                    c = control_design(test_coeffs, lambda_lims)  
                except:
                    print("No solution found")
                    c = control_design(test_coeffs_prev, lambda_lims) 
                    test_coeffs = test_coeffs_prev
                    
                test_coeffs_prev = test_coeffs
                dim = len(test_coeffs)-1

                if dim != dim_previous:
                    y = [np.zeros(f.dom.shape) for _ in range(dim)]
                    dim_previous = dim

            e = - f.gradient(x[...,k], k*f.time.t_s)
      
            y = y[1:] + [-sum([test_coeffs[i]*y[i]/test_coeffs[dim] for i in range(dim)]) + e/test_coeffs[dim]]
            
            x[...,k+1] = sum([c[i]*y[i] for i in range(dim)])
            x[...,k+1] = projection(x[...,k+1],sets)
            x_trial[...,k] = x[...,k+1].reshape(-1)

            e_hat, P, e_k_RLS = RLS_func(e_hat, P, f_factor2, k, x_trial, order)
            e_k_norm = window_error(e_hat, e_k_RLS, x_trial, k-1, order, win_size=win_size2)

            if e_k_norm < e_k_best:
                e_k_best = e_k_norm
                e_hat_best = e_hat
                kbest = k

            if kbest + 20 == k:
                test_coeffs = np.append(e_hat_best[::-1], 1.0)

            if window_error(e_hat_best, 0, x_trial, k, len(e_hat_best), win_size=2) > window_error(e_hat_best, 0, x_trial, k-1, len(e_hat_best), win_size=9)*1e4 and k > krecomp + 20: 
                threshold_reached = False
                print(f"Reset sys id at {k}")
                order = 1
                order_prev = 1
                dim = 0
                dim_previous = -1
                y = [np.zeros(f.dom.shape) for _ in range(order)]
                e_hat = np.ones((order,1))
                e_threshold = e_threshold2
                P = np.eye(order) * 1
                c = np.ones(order+1)
                offset = k - 3
            
        if threshold_reached == False and (k != offset or k == 0):
            y_unc = x[...,k]
     
            y_unc = y_unc - step*f.gradient(y_unc, k*f.time.t_s)
            y_unc = projection(y_unc,sets)
            x[...,k+1] = y_unc

            x_trial[...,k] = x[...,k+1].reshape(-1)
            
            e_hat, P, e_k_RLS = RLS_func(e_hat, P, f_factor1, k, x_trial, order)
            e_k_norm = window_error(e_hat, e_k_RLS, x_trial, k-1, order, win_size=win_size1)
            
            if e_k_norm < e_threshold:
                e_hat_best = e_hat
                test_coeffs = np.append(e_hat_best[::-1], 1.0)
                threshold_reached = True
                e_k_best = e_threshold
                
    return x, test_coeffs

def check_constraints(x, sets):
    # x shape: (6, 1, 8641)
    num_dims, _, num_steps = x.shape

    violations = np.zeros((num_dims, num_steps), dtype=bool)
    # violations[d, t] = True if x[d,0,t] violates all intervals

    for d in range(num_dims):
        vals = x[d, 0, :]  # shape (8641,)
        intervals = sets[d]  # list of allowed intervals for dim d
        
        # For each time, check if val lies in any allowed interval
        inside_any = np.zeros(num_steps, dtype=bool)
        for (low, high) in intervals:
            inside_any = inside_any | ((vals >= low) & (vals <= high))
        
        violations[d, :] = ~inside_any  # True where value outside all intervals

    if violations.any():
        print("Some violations found:")
        for d in range(num_dims):
            times_viol = np.where(violations[d, :])[0]
            if len(times_viol) > 0:
                print(f"Dimension {d} violates constraints at time steps: {times_viol}")
    else:
        print("All values satisfy the constraints.")

    return violations


# Online gradient with compensation, designed using control theory
def projected_online_gradient_control(problem, c, sets, x_0=0):
    
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
        x[...,k+1] = projection(x[...,k+1],sets)
    return x

# Online gradient with compensation, designed using control theory
def anti_windup_projected_online_gradient_control(problem, c, sets, rho, x_0=0):
    
    f, b = problem["f"], problem["b"]
    if len(c) >= len(b): raise ValueError("The controller must be strictly proper (`c` of length strictly smaller than `b`).")
    
    # compute the coefficients for the output
    m = len(b)-1
    
    x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    x[...,0] = x_0
    y = [np.zeros(f.dom.shape) for _ in range(m)]
    unproj = np.zeros(f.dom.shape + (f.time.num_samples+1,))
    unproj[...,0] = x_0
    
    for k in range(f.time.num_samples):
                
        e = - f.gradient(x[...,k], k*f.time.t_s) + rho * (x[...,k] - unproj[...,k])
                
        y = y[1:] + [-sum([b[i]*y[i]/b[m] for i in range(m)]) + e/b[m]]

        x[...,k+1] = sum([c[i]*y[i] for i in range(m)])
        unproj[...,k+1] = x[...,k+1]
        x[...,k+1] = projection(x[...,k+1],sets)
    return x