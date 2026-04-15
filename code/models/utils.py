import torch
import torch.nn.functional as F

def compute_distance(x1, x2, distance_type='euclidean', eps=1e-5):
    """
    Compute pairwise distance between x1 and x2.

    Args:
        x1, x2: Tensors of shape (N, d)
        distance_type: 'euclidean' or 'hyperbolic'
        eps: Small value to avoid numerical instability in hyperbolic space

    Returns:
        dist: Tensor of shape (N,), each entry is the distance between x1[i] and x2[i]
    """
    if distance_type == 'euclidean':
        return F.pairwise_distance(x1, x2, p=2)

    elif distance_type == 'hyperbolic':
        x1_norm = torch.clamp(torch.norm(x1, dim=-1, keepdim=True), max=1 - eps)
        x2_norm = torch.clamp(torch.norm(x2, dim=-1, keepdim=True), max=1 - eps)
        diff_norm = torch.norm(x1 - x2, dim=-1)
        denominator = (1 - x1_norm.pow(2)) * (1 - x2_norm.pow(2))
        return torch.acosh(1 + 2 * diff_norm.pow(2) / denominator.squeeze(-1))

    else:
        raise ValueError(f"Unsupported distance_type: {distance_type}")
    
def triplet_ranking_loss(anchor, positive, negative, margin=1.0, distance_type='euclidean'):
    """
    Triplet Ranking Loss

    Args:
        anchor: (N, d) tensor
        positive: (N, d) tensor
        negative: (N, d) tensor
        margin: margin for separation
        distance_type: 'euclidean' or 'hyperbolic'

    Returns:
        loss: scalar tensor
    """
    anchor = anchor.float()
    positive = positive.float()
    negative = negative.float()

    dist_pos = compute_distance(anchor, positive, distance_type=distance_type)
    dist_neg = compute_distance(anchor, negative, distance_type=distance_type)

    loss = F.relu(dist_pos - dist_neg + margin).mean()
    return loss