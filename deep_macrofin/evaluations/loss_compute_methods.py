from enum import Enum

import torch


class LossReductionMethod(str, Enum):
    MSE = "MSE" # mean squared error
    MAE = "MAE" # mean absolute error
    NONE = "None" # no reduction

LOSS_REDUCTION_MAP = {
    LossReductionMethod.MSE: lambda x: torch.mean(torch.square(x)),
    LossReductionMethod.MAE: lambda x: torch.mean(torch.abs(x)),
    LossReductionMethod.NONE: lambda x: x,
}