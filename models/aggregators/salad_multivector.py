import torch
import torch.nn as nn
import torch.nn.functional as F

from .salad import get_matching_probs


class SALAD_Multivector(nn.Module):
    """
    SALAD_Multivector (Sinkhorn Algorithm for Locally Aggregated Descriptors).

    Variant of SALAD that outputs m+1 separate L2-normalized vectors
    (1 global scene token + m cluster descriptors) instead of one concatenated vector.

    All vectors are projected to the same shared dimension D for MaxSim scoring.

    Attributes:
        num_channels (int): Number of input channels from backbone (d).
        num_clusters (int): Number of clusters (m).
        cluster_dim (int): Shared output dimension for all vectors (D).
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        dropout=0.3,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim

        if dropout > 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = nn.Identity()

        # MLP for global scene token — projects to cluster_dim
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.cluster_dim),
        )
        # MLP for local features f_i — projects to cluster_dim
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Args:
            x (tuple): (features, token)
                features (torch.Tensor): [B, C, H // 14, W // 14]
                token (torch.Tensor): [B, C]

        Returns:
            descriptors (torch.Tensor): [B, m+1, D] — m+1 L2-normalized vectors
                where vector 0 is the global scene token and vectors 1..m are cluster descriptors
        """
        x, t = x  # Extract features and token

        f = self.cluster_features(x).flatten(2)  # [B, D, N] where N = H*W
        p = self.score(x).flatten(2)  # [B, m, N]
        t = self.token_features(t)  # [B, D]

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dust_bin, num_iters=3, reg=0.5)
        p = torch.exp(p)
        # Remove dustbin row
        p = p[:, :-1, :]  # [B, m, N]

        # Weighted aggregation per cluster
        # p: [B, m, N], f: [B, D, N]
        # For each cluster k, compute: sum_n(p[k,n] * f[:,n]) → [B, D, m]
        cluster_embs = torch.einsum("bmn,bdn->bmd", p, f)  # [B, m, D]

        # L2-normalize each cluster embedding
        cluster_embs = F.normalize(cluster_embs, p=2, dim=-1)  # [B, m, D]

        # L2-normalize global token
        global_emb = F.normalize(t, p=2, dim=-1)  # [B, D]

        # Stack: [global, cluster_1, ..., cluster_m] → [B, m+1, D]
        descriptors = torch.cat(
            [
                global_emb.unsqueeze(1),  # [B, 1, D]
                cluster_embs,  # [B, m, D]
            ],
            dim=1,
        )  # [B, m+1, D]

        return descriptors

    def forward_debug(self, x):
        """
        Same as forward(), but also returns the Sinkhorn assignment probabilities
        for visualization — including the dustbin row.

        The Sinkhorn OT plan enforces:
          - Each patch n has total mass 1:  Σ_{k=0..m} p[k,n] = 1  (m clusters + dustbin)
          - Each cluster k gets equal mass: Σ_n p[k,n] = N/(m+1)

        So p[k,n] = fraction of patch n's mass assigned to cluster k.

        Args:
            x (tuple): (features, token) — same as forward()

        Returns:
            descriptors (torch.Tensor): [B, m+1, D]
            p_spatial   (torch.Tensor): [B, m, H_p, W_p]
                Per-cluster assignment probability for each patch.
                Values in [0,1]; summing p_spatial + p_dustbin over the cluster
                dimension gives ~1 per patch.
            p_dustbin   (torch.Tensor): [B, H_p, W_p]
                Fraction of each patch's mass that was sent to the dustbin
                (i.e. deemed uninformative by the model).
        """
        x, t = x
        B = x.shape[0]
        H_p, W_p = x.shape[2], x.shape[3]

        f = self.cluster_features(x).flatten(2)  # [B, D, N]
        p = self.score(x).flatten(2)  # [B, m, N]
        t = self.token_features(t)  # [B, D]

        # Sinkhorn — output is [B, m+1, N]  (last row = dustbin)
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)

        # Capture dustbin BEFORE removing it
        p_dustbin = p[:, -1, :].reshape(B, H_p, W_p)  # [B, H_p, W_p]

        p = p[:, :-1, :]  # remove dustbin  [B, m, N]

        # Reshape cluster assignments to spatial grid
        p_spatial = p.reshape(B, self.num_clusters, H_p, W_p)  # [B, m, H_p, W_p]

        # Weighted aggregation
        cluster_embs = torch.einsum("bmn,bdn->bmd", p, f)
        cluster_embs = F.normalize(cluster_embs, p=2, dim=-1)
        global_emb = F.normalize(t, p=2, dim=-1)

        descriptors = torch.cat(
            [
                global_emb.unsqueeze(1),
                cluster_embs,
            ],
            dim=1,
        )  # [B, m+1, D]

        return descriptors, p_spatial, p_dustbin
