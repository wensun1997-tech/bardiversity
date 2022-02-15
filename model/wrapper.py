import torch
from torch import Tensor
import torch.nn as nn


class BARStructuredWrapper(nn.Module):
    r""" 
    Module wrapper for structured pruning.
    Implementation for "Structured Pruning of Neural Networks with Budget-Aware Regularization, `Link_`".

    Args:
        module (nn.Module): Base module.
        alpha (float): Default: 0.
        beta (float): Default: 0.667.
        gamma (float): Default: -0.1
        zeta (float): Default: 1.1.

    Examples::
        >>> module = nn.BatchNorm2d(3)
        >>> wrapper = BARStructuredWrapper(module)
        >>> data = torch.randn((2, 3, 3, 4))
        >>> output = wrapper(data)

    .. _Link:
        https://arxiv.org/abs/1811.09332
    """
    def __init__(self, module: nn.Module, alpha: float = 0., beta: float = 0.667, 
            gamma: float = -0.1, zeta: float = 1.1) -> None:
        super(BARStructuredWrapper, self).__init__()
        # the wrapped module
        self.module = module
        
        # pruning hyper-parameters
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.zeta = zeta
        
        # to be calculated in the first feed-forward
        self.log_alpha = nn.Parameter(torch.rand(0), requires_grad = True)
        self.area = None  # the computation overhead

        # whether calculate the mask with randomness, used for pruning training process
        # set to be ``False`` in the validation process
        self.stochastic = True

        self.to_initialize = True  # whether has be initialized

        self.hard_mask = None  # generate the binary mask after pruning training

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        output = self.module(input, *args, **kwargs)
        if self.to_initialize:
            # calculate ``area = H * W``
            self.area = output.size(2) * output.size(3)
            # initialize parameters for gates
            self.log_alpha.data = torch.rand(output.size(1)) * 0.01 + self.alpha
            self.to_initialize = False

        if self.hard_mask is None:
            z = self.cal_mask(self.stochastic).to(input.device)
        else:
            z = self.hard_mask.to(input.device)

        output *= z[None, :, None, None]
        return output
            
    def cal_mask(self, stochastic: bool) -> Tensor:
        r"""
        Calculate the mask.

        Args:
            stochastic (bool): Whether calculate the mask with randomness, used for pruning 
                               training process, set to be ``False`` in the validation process.

        Returns:
            z (Tensor): The mask.
        """
        assert not self.to_initialize, "Please feed-forward for one-step before."
        nchannels = len(self.log_alpha)
        if stochastic:
            u = torch.rand(nchannels).requires_grad_(False).to(self.log_alpha)
            s = torch.sigmoid((torch.log(u) - torch.log(1. - u) + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, min = 0., max = 1.)
        return z

    def hard_prune(self) -> Tensor:
        r"""
        Hard prune the network to get the final binary mask.
        """
        s = torch.sigmoid(self.log_alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, min = 0., max = 1.)
        self.hard_mask = (z > 0.).long()
        self.log_alpha = None
        return self.hard_mask

    def train(self, mode: bool = True) -> nn.Module:
        r"""
        Sets the module in training mode.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            nn.Module: self.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.stochastic = mode
        return self

    def eval(self) -> nn.Module:
        r"""
        Sets the module in evaluation mode.

        Returns:
            nn.Module: self.
        """
        return self.train(False)

    @property
    def computation_overhead(self) -> float:
        r"""
        Get the computation overhead.

        Returns:
            area (float): The computation overhead is defined as the area in BAR.
        """
        return self.area
