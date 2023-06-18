import random
import torch

def seed_everything(seed=42):
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        for t, p in enumerate(self.params):
            if p.grad is not None:
                self.m[t] = self.beta1 * self.m[t] + (1 - self.beta1) * p.grad
                self.v[t] = self.beta2 * self.v[t] + (1 - self.beta2) * p.grad**2
                m_hat = self.m[t] / (1 - self.beta1**(t + 1))
                v_hat = self.v[t] / (1 - self.beta2**(t + 1))
                p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)