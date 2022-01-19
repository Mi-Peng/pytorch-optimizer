'''
Unofficial Implementation of LARS in Pytorch

Ref:
    - Paper: https://arxiv.org/pdf/1708.03888.pdf (`Large batch training of Convolutional Networks`)
    - Paper: https://arxiv.org/pdf/1904.00962.pdf (`Large batch optimization for deep learning: training BERT in 76 minutes`)
    - Source Code of Lamb: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py#L143
    - timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lars.py
    - NIVIDA apex: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
'''
import torch
from torch.optim.optimizer import Optimizer

class LARS(Optimizer):
    '''
    Paper: `Large batch training of Convolutional Networks`(https://arxiv.org/pdf/1708.03888.pdf)

    Args:
        - params: parameters to optimize or dicts.
        - lr(float): learning rate(default: 1.0).
        - momentum(float): momentum factor(default: 0).
        - weight_decay(float): weight decay(default: 0).
        - dampening(float): dampening for momentum(default: 0).
        - nesterov(bool): enables Nesterov momentum(default: False).
        - trust_coeff(float): trust coefficient for computing adaptive lr(default: 0.001).
        - eps(float): eps for division denominator(default: 1e-8).
        - trust_clip(bool): enable LARS adaptive lr clipping(default: False).
    '''
    def __init__(
        self, 
        params, 
        lr: float=1.0,
        momentum: float=0,
        weight_decay: float=0,
        dampening: float=0,
        nesterov: bool=False,
        trust_coeff: float=0.001,
        eps: float=1e-8,
        trust_clip: bool=False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <=0 or dampening != 0):
            raise ValueError("Nesterov requires a non-zero momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            trust_coeff=trust_coeff,
            eps=eps,
            trust_clip=trust_clip,
        )
        super.__init__(params, defaults)
    
    def __setstate__(self, state: dict):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        one_tensor = torch.tensor(1.0, device=self.param_groups[0]['params'][0].device)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            trust_coeff = group['trust_coeff']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad        
                # compute local learning rate 
                if weight_decay != 0:
                    w_norm = torch.norm(p, 2)
                    g_norm = torch.norm(p.grad.data, 2)
                    adaptive_lr = trust_coeff * w_norm / (g_norm + w_norm * weight_decay + eps)
                    # handle when w_norm ==0 or g_norm == 0, LARS degenerates to normal SGD
                    adaptive_lr = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, adaptive_lr, one_tensor),
                        one_tensor,
                    )
                    # clip the local learning rate: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py#L97
                    if group['trust_clip']:
                        adaptive_lr = torch.minimum(adaptive_lr/group['lr'], one_tensor)
                    grad.add_(p, alpha=weight_decay)
                    grad.mul_(adaptive_lr)

                # update as SGD: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha= 1.0 - dampening)

                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                p.add_(grad, alpha=-group['lr'])
        return loss        