'''
Unofficial Implementation of LAMB in Pytorch

Ref:
    - Paper: https://arxiv.org/pdf/1904.00962.pdf (`Large batch optimization for deep learning: training BERT in 76 minutes`)
    - Source Code of Lamb: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py#L143
    - timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lamb.py
    - NVIDIA: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py#L170
'''
from typing import Tuple
import torch
from torch.optim.optimizer import Optimizer

class LAMB(Optimizer):
    '''
    Paper: `Large batch optimization for deep learning: training BERT in 76 minutes`(https://arxiv.org/pdf/1904.00962.pdf)

    Args:
        - params: parameters to optimize or dicts.
        - lr(float): learning rate(default: 1.0).
        - betas(Tuple[float, float]): coefficients used for computing
            running averages of gradient and its square(default: (0.9, 0.999)).
        - weight_decay(float): weight decay(default: 0).
        - eps(float): eps for division denominator(default: 1e-8).
    '''
    def __init__(
        self,
        params,
        lr: float=1.0,
        betas: Tuple[float, float]=(0.9, 0.999),
        weight_decay: float=0,
        eps: float=1e-8,
    ):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError("Invalid beta1 value: {}".format(betas[0]))
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError("Invalid beta2 value: {}".format(betas[1]))    
        if not weight_decay >= 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')
                
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 1
                    # m: Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p.data)
                    # v: Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p.data)
                # update m & v:
                m, v = state['m'], state['v']
                m = m * beta1 + (1 - beta1) * grad
                v = v * beta2 + (1 - beta2) * (grad * grad)
                m = m / (1 - beta1 ** state['step'])
                v = v / (1 - beta2 ** state['step'])
                state['m'], state['v'] = m, v
                state['step'] += 1

                # cal ratio:
                update = m / (v.sqrt() + eps)
                if weight_decay != 0:
                    update = update + weight_decay * p
                
                # adaptive lr:
                w_norm = torch.norm(p, 2)
                g_norm = torch.norm(update, 2)
                ratio = (w_norm == 0.0) * 1.0 + (w_norm != 0.0) * ratio
                ratio = (g_norm == 0.0) * 1.0 + (g_norm != 0.0) * ratio
                ratio = ratio.float()

                # optimize params
                p = p - lr * ratio * update
        return loss