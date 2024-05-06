from typing import Tuple, Union, Optional

import torch
import numpy as np
import numpy.typing as npt


def batch_inv_R_t(R: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    inv_R = R.permute(0, 2, 1).contiguous()    
    inv_t = -torch.bmm(inv_R, t.unsqueeze(-1)).squeeze(-1) if t is not None else None
    return inv_R, inv_t

def inv_R_t(R: npt.NDArray, t: Optional[npt.NDArray] = None) -> Tuple[npt.NDArray, Union[npt.NDArray, None]]:
    inv_R = R.T
    inv_t = -np.matmul(inv_R, t) if t is not None else None
    return inv_R, inv_t