import numpy as np
ran = np.random.default_rng()
from tvopt import costs, sets

class use_case_tester_dis(costs.Cost):
    
    def __init__(self, G, H, w, y_ref, beta, U1 , U2, U3, U4, U5, U6, t_s=1):
        
        d, num_samples = G.shape[1], w[0].shape[0]
        super().__init__(sets.R(d, 1), sets.T(t_s, t_max=t_s*num_samples))
        self.beta = beta
        self.H, self.w, self.y_ref, self.G = H, w, y_ref, G
        self.U1, self.U2, self.U3, self.U4, self.U5, self.U6 = U1, U2, U3, U4, U5, U6
        self.smooth = 2
    
    def function(self, x, t):
        k = round(t / self.time.t_s)
        y = self.G @ x + (self.H @ self.w[:,k]).reshape(self.w.shape[0],1)
        diff = y - (self.y_ref[:,k].reshape(self.w.shape[0],1))
        base = 0.5 * self.beta * diff.T @ diff
        
        if t < 2160 or (t > 4320 and t < 6480):
            u = x.T.dot(0.5*self.U1.dot(x) + self.U2) + self.U3
        else:
            u = x.T.dot(0.5*self.U4.dot(x) + self.U5) + self.U6
        return base + u

        
    def gradient(self, x, t):
        k = round(t / self.time.t_s)
        y = self.G @ x + (self.H @ self.w[:,k]).reshape(self.w.shape[0],1)
        diff = y - (self.y_ref[:,k].reshape(self.w.shape[0],1))
        grad = self.beta * self.G.T @ diff

        if t < 2160 or (t > 4320 and t < 6480):
            grad += self.U1.dot(x) + self.U2
        else:
            grad += self.U4.dot(x) + self.U5
        return grad
    
class use_case_tester(costs.Cost):
    
    def __init__(self, G, H, w, y_ref, beta, t_s=1):
        
        d, num_samples = G.shape[1], w[0].shape[0]
        super().__init__(sets.R(d, 1), sets.T(t_s, t_max=t_s*num_samples))
        self.beta = beta
        self.H, self.w, self.y_ref, self.G = H, w, y_ref, G
        
        self.smooth = 2
    
    def function(self, x, t):
        k = round(t / self.time.t_s)
        y = self.G @ x + (self.H @ self.w[:,k]).reshape(self.w.shape[0],1)
        diff = y - (self.y_ref[:,k].reshape(self.w.shape[0],1))
        return 0.5 * self.beta * diff.T @ diff

        
    def gradient(self, x, t):
        k = round(t / self.time.t_s)
        y = self.G @ x + (self.H @ self.w[:,k]).reshape(self.w.shape[0],1)
        diff = y - (self.y_ref[:,k].reshape(self.w.shape[0],1))
        return self.beta * self.G.T @ diff