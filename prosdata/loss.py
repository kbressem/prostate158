import monai
from .utils import load_config


def get_loss(config: dict): 
    """Create a loss function of `type` with specific keyword arguments from config.
    Example: 
        
        config.loss
        >>> {'DiceCELoss': {'include_background': False, 'softmax': True, 'to_onehot_y': True}}

        get_loss(config)
        >>> DiceCELoss(
        >>>   (dice): DiceLoss()
        >>>   (cross_entropy): CrossEntropyLoss()
        >>> )
    
    """
    loss_type = list(config.loss.keys())[0]
    loss_config = config.loss[loss_type]
    loss_fun =  getattr(monai.losses, loss_type)
    loss = loss_fun(**loss_config)
    return loss