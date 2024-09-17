from enum import Enum

import torch


class LossReductionMethod(str, Enum):
    MSE = "MSE" # mean squared error
    MAE = "MAE" # mean absolute error
    SSE = "SSE" # sum squared error
    SAE = "SAE" # sum absolute error
    NONE = "None" # no reduction

LOSS_REDUCTION_MAP = {
    LossReductionMethod.MSE: lambda x: torch.mean(torch.square(x)),
    LossReductionMethod.MAE: lambda x: torch.mean(torch.abs(x)),
    LossReductionMethod.SSE: lambda x: torch.sum(torch.square(x)),
    LossReductionMethod.SAE: lambda x: torch.sum(torch.absolute(x)),
    LossReductionMethod.NONE: lambda x: x,
}