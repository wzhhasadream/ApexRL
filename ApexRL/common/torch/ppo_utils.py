import torch


def diagonal_gaussian_kl(
    mean_current: torch.Tensor,
    std_current: torch.Tensor,
    mean_old: torch.Tensor,
    std_old: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(old || current) for diagonal Gaussian policies."""
    kl = (
        torch.log(std_current / std_old)
        + (std_old.square() + (mean_old - mean_current).square())
        / (2.0 * std_current.square())
        - 0.5
    )
    return kl.sum(dim=-1, keepdim=True)


def categorical_kl(
    logits_current: torch.Tensor,
    logits_old: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(old || current) for categorical policies from logits."""
    log_probs_current = torch.log_softmax(logits_current, dim=-1)
    log_probs_old = torch.log_softmax(logits_old, dim=-1)
    probs_old = log_probs_old.exp()
    return (probs_old * (log_probs_old - log_probs_current)).sum(dim=-1, keepdim=True)


def adapt_lr(
    lr: float,
    kl: torch.Tensor,
    desired_kl: float = 0.01,
    lr_min: float = 1e-5,
    lr_max: float = 1e-2,
    factor: float = 1.5,
) -> float:
    kl_value = float(kl.detach())
    if kl_value > 2.0 * desired_kl:
        return max(lr / factor, lr_min)
    if 0.0 < kl_value < 0.5 * desired_kl:
        return min(lr * factor, lr_max)
    return lr
