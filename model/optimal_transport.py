import torch
import torch.nn as nn

def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 5, slack: bool = False) -> torch.Tensor:
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        if log_alpha.ndim == 3:
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = log_alpha_padded.squeeze(dim=1)
        else:
            log_alpha_padded = zero_pad(log_alpha)
        
        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[..., :-1, :] - torch.logsumexp(log_alpha_padded[..., :-1, :], dim=-1, keepdim=True),
                    log_alpha_padded[..., -1:, :]), dim=-2)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[..., :, :-1] - torch.logsumexp(log_alpha_padded[..., :, :-1], dim=-2, keepdim=True),
                    log_alpha_padded[..., :, -1:]), dim=-1)

        log_alpha = log_alpha_padded[..., :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

    return log_alpha