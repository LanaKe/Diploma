import importlib
import math
import warnings
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch import _VF, sym_int as _sym_int, Tensor
from torch._C import _add_docstr, _infer_size
from torch._jit_internal import (
    _overload,
    boolean_dispatch,
    BroadcastingList1,
    BroadcastingList2,
    BroadcastingList3,
)
from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notes
from torch.nn import _reduction as _Reduction, grad  # noqa: F401
from torch.nn.modules.utils import _list_with_default, _pair, _single, _triple
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)

if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor:
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None) -> Tensor

    Measures the element-wise mean squared error, with optional weighting.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        size_average (bool, optional): Deprecated (use reduction).
        reduce (bool, optional): Deprecated (use reduction).
        reduction (str, optional): Specifies the reduction to apply to the output:
                                'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                'sum': the output will be summed. 'none': no reduction will be applied.
                                Default: 'mean'.
        weight (Tensor, optional): Weights for each sample. Default: None.

    Returns:
        Tensor: Mean Squared Error loss (optionally weighted).
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            mse_loss,
            (input, target, weight),
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            weight=weight,
        )

    if not (target.size() == input.size()):
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)

    if weight is not None:
        if weight.size() != input.size():
            raise ValueError("Weights and input must have the same size.")

        # Perform weighted MSE loss manually
        squared_errors = torch.pow(expanded_input - expanded_target, 2)
        weighted_squared_errors = squared_errors * weight

        if reduction == "none":
            return weighted_squared_errors
        elif reduction == "sum":
            return torch.sum(weighted_squared_errors)
        elif reduction == "mean":
            return torch.sum(weighted_squared_errors) / torch.sum(weight)
        else:
            raise ValueError(
                f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
            )
    else:
        return torch._C._nn.mse_loss(
            expanded_input, expanded_target, _Reduction.get_enum(reduction)
        )