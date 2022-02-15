import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from extorch.nn.utils import net_device

from model.wrapper import BARStructuredWrapper


def exp_progress_fn(p: float, a: float = 4.) -> float:
    c = 1. - np.exp(-a)
    exp_progress = 1. - np.exp(-a * p)
    return exp_progress / c


def sigmoid_progress_fn(p: float, a: float) -> float:
    b = 1. / (1. + np.exp(a * 0.5))
    sigmoid_progress = 1. / (1. + np.exp(a * (0.5 - p)))
    sigmoid_progress = (sigmoid_progress - b) / (1. - 2. * b)
    return sigmoid_progress


class DistillationLoss(nn.Module):
    r"""
    Distillation objective.

    Args:
        T (float): The temperature.
        alpha (float): The coefficient to controll the trade-off between distillation loss and origin loss.
    """
    def __init__(self, T: float, alpha: float) -> None:
        super(DistillationLoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, output: Tensor, target: Tensor, label: Tensor) -> Tensor:
        r"""
        Args:
            output (Tensor): Output of the network to be trained.
            target (Tensor): Output of the teacher network.
            label (Tensor): Label of the input.

        Returns:
            Tensor: The calculated loss.
        """
        p = F.softmax(target / self.T, dim = 1)
        log_q = F.log_softmax(output / self.T, dim = 1)
        entropy = - torch.sum(p * log_q, dim = 1)
        kl = F.kl_div(log_q, p, reduction = "mean")
        loss = torch.mean(entropy + kl)
        return self.alpha * self.T ** 2 * loss + \
                F.cross_entropy(output, label) * (1 - self.alpha)


class BudgetLoss(nn.Module):
    r"""
    Budget loss for BAR.
    """
    def __init__(self) -> Tensor:
        super(BudgetLoss, self).__init__()

    def forward(self, net: nn.Module) -> Tensor:
        r"""
        Calculate the budget loss.

        Args:
            net (nn.Module): The network to be pruned.

        Returns:
            loss (Tensor): The budget loss.
        """
        loss = torch.zeros(1).to(net_device(net))
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                # Probability of being alive for each feature map
                alive_probability = torch.sigmoid(
                        m.log_alpha - m.beta * np.log(-m.gamma / m.zeta))
                loss += torch.sum(alive_probability) * m.computation_overhead
        return loss


class BARStructuredLoss(nn.Module):
    r"""
    Objective of Budget-Aware Regularization Structured Pruning.

    Args:
        budget (float): The budget.
        epochs (int): Total pruning epochs.
        progress_func (str): Type of progress function ("sigmoid" or "exp"). Default: "sigmoid".
        _lambda (float): Coefficient for trade-off of sparsity loss term. Default: 1e-5.
        distillation_temperature (float): Knowledge Distillation temperature. Default: 4.
        distillation_alpha (float): Knowledge Distillation alpha. Default: 0.9.
        tolerance (float): Default: 0.01.
        margin (float): Parameter a in Eq. 5 of the paper. Default: 0.0001.
        sigmoid_a (float): Slope parameter of sigmoidal progress function. Default: 10.
        upper_bound (float): Default: 1e10.
    """
    def __init__(self, budget: float, epochs: int, progress_func: str = "sigmoid", 
            _lambda: float = 1e-5, distillation_temperature: float = 4., 
            distillation_alpha: float = 0.9, tolerance: float = 0.01, margin: float = 1e-4,
            sigmoid_a: float = 10., upper_bound: float = 1e10) -> None:
        super(BARStructuredLoss, self).__init__()
        self.budget = budget
        self.epochs = epochs

        self.progress_func = progress_func
        self.sigmoid_a = sigmoid_a
        self.tolerance = tolerance
        self.upper_bound = upper_bound
        
        self._lambda = _lambda
        self.margin = margin

        self.classification_criterion = DistillationLoss(distillation_temperature, distillation_alpha)
        self.budget_criterion = BudgetLoss()

        self._origin_overhead = None

    def forward(self, mask, input: Tensor, output: Tensor, target: Tensor,
            net: nn.Module, teacher: nn.Module, current_epoch_fraction: float) -> Tensor:
        r"""
        Calculate the objective.

        Args:
            input (Tensor): Input image.
            output (Tensor): Output logit.
            target (Tensor): Label of the image,
            net (nn.Module): The network to be updated.
            teacher (nn.Module): Teacher network for distillation.
            current_epoch_fraction (float): Current epoch fraction.

        Returns:
            loss (Tensor): The loss.
        """
        # Step 1: Calculate the cross-entropy loss and distillation loss.
        with torch.no_grad():
            teacher_output = teacher(input)
        classification_loss = self.classification_criterion(output, teacher_output, target)

        # Step 2: Calculate the budget loss.
        budget_loss = self.budget_criterion(net)

        # Step 3: Calculate the coefficient of the budget loss.
        current_overhead = self.current_overhead(net)
        origin_overhead = self.origin_overhead(net)
        tolerant_overhead = (1. + self.tolerance) * origin_overhead
        
        p = current_epoch_fraction / self.epochs
        if self.progress_func == "sigmoid":
            p = sigmoid_progress_fn(p, self.sigmoid_a)
        elif self.progress_func == "exp":
            p = exp_progress_fn(p)
        
        current_budget = (1 - p) * tolerant_overhead + p * self.budget * origin_overhead

        margin = tolerant_overhead * self.margin
        lower_bound = self.budget * origin_overhead - margin
        budget_respect = (current_overhead - lower_bound) / (current_budget - lower_bound)
        budget_respect = max(budget_respect, 0.)
        
        if budget_respect < 1.:
            lamb_mult = min(budget_respect ** 2 / (1. - budget_respect), self.upper_bound)
        else:
            lamb_mult = self.upper_bound

        # Step 4: Compute diversity.
        mask_all = torch.tensor([])
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                z = m.cal_mask(stochastic = False)
                mask_all = torch.cat([mask_all, z])

        diversity = 0
        for i in range(len(mask)):
            mask_soft = F.softmax(mask[i])
            log_mask_all = F.log_softmax(mask_all)
            diversity += F.kl_div(log_mask_all, mask_soft)
        diversity = diversity / len(mask)

        # Step 5: Combine the objectives.
        loss = classification_loss + self._lambda / len(input) * lamb_mult * budget_loss - 1e6 * diversity
        return loss

    def current_overhead(self, net: nn.Module) -> float:
        r"""
        Calculate the computation overhead after pruning.

        Args:
            net (nn.Module): The network to be calculated.

        Returns:
            overhead (float): The computation overhead after pruning.
        """
        overhead = 0
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                z = m.cal_mask(stochastic = False)
                overhead += m.computation_overhead * (z > 0.).long().sum().item()
        return overhead

    def origin_overhead(self, net: nn.Module) -> float:
        r"""
        Calculate the origin computation overhead before pruning.

        Args:
            net (nn.Module): The network to be calculated.

        Returns:
            overhead (float): The origin computation overhead before pruning.
        """
        if self._origin_overhead:
            return self._origin_overhead

        self._origin_overhead = 0
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                nchannels = len(m.log_alpha)
                self._origin_overhead += m.computation_overhead * nchannels
        return self._origin_overhead

    def sparsity_ratio(self, net: nn.Module) -> float:
        r"""
        Calculate the spartial ratio.
        
        Args:
            net (nn.Module): The network to be calculated.

        Returns:
            sparsity_ratio (float): The spartial ratio.
        """
        current_overhead = self.current_overhead(net)
        origin_overhead = self.origin_overhead(net)
        sparsity_ratio = current_overhead / origin_overhead
        return sparsity_ratio
