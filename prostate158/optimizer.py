import torch
import monai
from .utils import load_config


def get_optimizer(model: torch.nn.Module, 
                  config: dict): 
    """Create an optimizer of `type` with specific keyword arguments from config.
    Example: 
        
        config.optimizer
        >>> {'Novograd': {'lr': 0.001, 'weight_decay': 0.01}}

        get_optimizer(model, config)
        >>> Novograd (
        >>> Parameter Group 0
        >>>     amsgrad: False
        >>>     betas: (0.9, 0.999)
        >>>     eps: 1e-08
        >>>     grad_averaging: False
        >>>     lr: 0.0001
        >>>     weight_decay: 0.001
        >>> )
    
    """
    optimizer_type = list(config.optimizer.keys())[0]
    opt_config = config.optimizer[optimizer_type]
    if hasattr(torch.optim, optimizer_type): 
        optimizer_fun = getattr(torch.optim, optimizer_type)
    elif hasattr(monai.optimizers, optimizer_type): 
        optimizer_fun = getattr(monai.optimizers, optimizer_type)
    else: 
        raise ValueError(f'Optimizer {optimizer_type} not found')
    optimizer = optimizer_fun(model.parameters(), **opt_config)
    return optimizer


