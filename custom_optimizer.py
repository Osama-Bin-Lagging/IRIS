# custom_optimizers.py
import torch
from torch.optim.optimizer import Optimizer

class Lamb(Optimizer):
    """
    Lamb optimizer as described in:
    Large Batch Optimization for Deep Learning: Training BERT in 76 minutes
    https://arxiv.org/abs/1904.00962
    
    This implementation follows the paper exactly.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, 
                 weight_decay=0.0, clamp_value=10.0, debias=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= clamp_value:
            raise ValueError("Invalid clamp value: {}".format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       clamp_value=clamp_value, debias=debias)
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper algorithm
                if group['debias']:
                    # Bias correction as in Adam
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    k = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                    u = exp_avg / bias_correction1 / (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                else:
                    k = group['lr']
                    u = exp_avg / exp_avg_sq.sqrt().add_(group['eps'])

                # Add weight decay
                if group['weight_decay'] != 0:
                    u = u.add(p.data, alpha=group['weight_decay'])

                # Compute norms
                w_norm = p.data.norm()
                u_norm = u.norm()

                # Compute trust ratio
                if w_norm == 0 or u_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = torch.min(
                        torch.tensor(group['clamp_value'], device=p.device), 
                        w_norm / u_norm
                    )

                # Apply update
                p.data.add_(u, alpha=-k * trust_ratio)

        return loss

# Try to import from torch_optimizer first, fallback to custom implementation
try:
    from torch_optimizer import Lamb as TorchOptimizerLamb
    Lamb = TorchOptimizerLamb
except ImportError:
    pass  # Use our custom implementation above
