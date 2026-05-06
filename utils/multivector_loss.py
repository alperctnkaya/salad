import torch
import torch.nn as nn
import torch.nn.functional as F


def maxsim(Q, R):
    """
    Compute MaxSim scores between two sets of multivector descriptors.

    MaxSim(Q, R) = Σ_i max_j cos(Q_i, R_j)

    Args:
        Q (torch.Tensor): [Bq, V, D] query descriptors (L2-normalized)
        R (torch.Tensor): [Br, V, D] reference descriptors (L2-normalized)

    Returns:
        scores (torch.Tensor): [Bq, Br] MaxSim similarity matrix
    """
    # sim: [Bq, Vq, Br, Vr]
    sim = torch.einsum("qid,rjd->qirj", Q, R)
    # For each query vector, max over reference vectors: [Bq, Vq, Br]
    max_sim = sim.max(dim=-1).values
    # Sum over query vectors: [Bq, Br]
    return max_sim.sum(dim=1)


class MaxSimMSLoss(nn.Module):
    """
    Multi-Similarity Loss operating on MaxSim similarity scores.

    Adapts the proven MS loss (used by original SALAD) to work with
    multivector MaxSim scores instead of single-vector cosine similarity.

    Includes built-in hard pair mining (no separate miner needed).

    MS Loss:
        L = (1/α) * log(1 + Σ_{pos} exp(-α(S_ij - λ)))
          + (1/β) * log(1 + Σ_{neg} exp( β(S_ij - λ)))

    where S_ij = normalized_maxsim(i, j)

    Args:
        alpha (float): Weight for positive pairs (default 2.0)
        beta (float): Weight for negative pairs (default 50.0)
        base (float): Margin λ (default 0.5)
        mining_epsilon (float): Margin for hard pair mining (default 0.1)
    """

    def __init__(self, alpha=1.0, beta=50.0, base=0.0, mining_epsilon=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.mining_epsilon = mining_epsilon

    def forward(self, descriptors, labels):
        """
        Args:
            descriptors (torch.Tensor): [B, V, D] L2-normalized multivector descriptors
            labels (torch.Tensor): [B] integer labels
        Returns:
            loss (torch.Tensor): scalar
            batch_acc (float): fraction of non-trivial samples
        """
        B, V, D = descriptors.shape
        device = descriptors.device

        # Normalized MaxSim similarity matrix in [-1, 1]
        sim_matrix = maxsim(descriptors, descriptors) / V  # [B, B]

        # Create masks
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = label_eq & ~self_mask
        neg_mask = ~label_eq

        # Initialize loss connected to the computational graph
        loss = descriptors.sum() * 0.0
        num_valid = 0

        for i in range(B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            pos_sim = sim_matrix[i, pos_idx]  # similarities to positives
            neg_sim = sim_matrix[i, neg_idx]  # similarities to negatives

            # Multi-Similarity Mining: keep hard pairs
            # Hard positives: pos pairs that have a negative more similar than them
            # Hard negatives: neg pairs that are more similar than some positive
            max_neg_sim = neg_sim.max()
            min_pos_sim = pos_sim.min()

            hard_pos_mask = pos_sim < max_neg_sim + self.mining_epsilon
            hard_neg_mask = neg_sim > min_pos_sim - self.mining_epsilon

            pos_sim = pos_sim[hard_pos_mask]
            neg_sim = neg_sim[hard_neg_mask]

            pos_loss = 0.0
            if len(pos_sim) > 0:
                pos_loss = (1.0 / self.alpha) * torch.log(
                    1 + torch.sum(torch.exp(-self.alpha * (pos_sim - self.base)))
                )

            neg_loss = 0.0
            if len(neg_sim) > 0:
                neg_loss = (1.0 / self.beta) * torch.log(
                    1 + torch.sum(torch.exp(self.beta * (neg_sim - self.base)))
                )

            loss += pos_loss + neg_loss
            num_valid += 1

        if num_valid > 0:
            loss = loss / num_valid

        batch_acc = 1.0 - (num_valid / B) if B > 0 else 0.0
        return loss, batch_acc

