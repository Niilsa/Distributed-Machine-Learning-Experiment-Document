import random
import torch

class BaseOptimizer():
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


class GDOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params, lr)
        
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad


class SGDOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params, lr)

    def step(self):
        for p in self.params:
            if p.grad is not None:
                random_index = random.randint(0, len(p.grad) - 1)
                random_grad = torch.zeros_like(p.grad)
                random_grad[random_index] = p.grad[random_index]
                p.data -= self.lr * random_grad
                            

class AdamOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = b1
        self.beta2 = b2
        self.eps = eps
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)