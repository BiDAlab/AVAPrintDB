import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TemporalAttentionPooling(nn.Module):
    """Applies temporal attention pooling over a sequence of embeddings.

    This module computes attention weights over the temporal dimension
    of an input tensor and returns a weighted average embedding.

    Attributes:
        query (torch.nn.Parameter): Learnable query vector of shape
            (embed_dim,) used to compute attention scores.
    """
    def __init__(self, embed_dim):
        """Initializes the TemporalAttentionPooling module.

        Args:
            embed_dim (int): Dimensionality of the input embeddings
                and the query vector.
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x, mask=None):
        """Applies temporal attention pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D), where
                B is the batch size, T is the number of timesteps,
                and D is the embedding dimension.
            mask (torch.Tensor, optional): Optional attention mask of
                shape (B, T). Positions with value 0 are masked out.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Pooled embedding of shape (B, D).
                - Attention weights of shape (B, T, 1).
        """
        scores = torch.matmul(x, self.query)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(x * att, dim=1), att


class SpatialTemporalTripletModel(nn.Module):
    """Graph convolutional network for spatial-temporal encoding with triplet loss.

    This model extracts embeddings from spatio-temporal graph sequences
    using GCN layers applied per-frame, followed by temporal attention
    pooling. It supports both triplet training (anchor, positive, negative)
    and single input prediction.

    Attributes:
        V (int): Number of graph nodes (facial landmarks) per frame.
        embed_dim (int): Dimensionality of the final embedding space.
        gcn1 (GCNConv): First graph convolutional layer.
        gcn2 (GCNConv): Second graph convolutional layer.
        gcn3 (GCNConv): Third graph convolutional layer projecting to
            embedding dimension.
        dropout (torch.nn.Dropout): Dropout module for regularization.
        temporal_pool (TemporalAttentionPooling): Module for temporal
            attention pooling.
        base_edge_index (torch.Tensor): Graph connectivity of shape (2, E),
            registered as a buffer.
    """

    def __init__(self, edge_index, num_nodes, in_dim, hidden_dim, embed_dim, dropout_p=0.3):
        """Initializes the SpatialTemporalTripletModel.

        Args:
            edge_index (torch.Tensor): Base graph edge index of shape (2, E).
            num_nodes (int): Number of nodes in the input graph.
            in_dim (int): Input feature dimension per node.
            hidden_dim (int): Hidden dimension for intermediate GCN layers.
            embed_dim (int): Output embedding dimension.
            dropout_p (float, optional): Dropout probability. Defaults to 0.3.
        """
        super().__init__()
        self.V = num_nodes
        self.embed_dim = embed_dim

        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        
        self.register_buffer('base_edge_index', edge_index)

    def spatial_encode(self, windows):
        """Encodes per-frame graph features with GCNs and mean pooling.

        Args:
            windows (torch.Tensor): Input tensor of shape (B, T, V, D),
                where B is the batch size, T the number of frames,
                V the number of nodes, and D the feature dimension.

        Returns:
            torch.Tensor: Encoded sequence of shape (B, T, embed_dim).
        """
        B, T, V, D = windows.shape
        BT = B * T
        E = self.base_edge_index.size(1) # type: ignore

        # Extnder edge_index para capas gcn
        offsets = (torch.arange(BT, device=windows.device) * V).repeat_interleave(E).unsqueeze(0)
        big_edge_index = self.base_edge_index.repeat(1, BT) + offsets  # type: ignore

        
        x = windows.view(BT * V, D)  

        # Gcns
        h = F.relu(self.gcn1(x, big_edge_index))
        h = self.dropout(h)
        h = F.relu(self.gcn2(h, big_edge_index))
        h = self.dropout(h)
        h = self.gcn3(h, big_edge_index)
        h = self.dropout(h)

        # pool independiente por grafo
        batch_vec = torch.arange(BT, device=windows.device).repeat_interleave(V)  

        emb = global_mean_pool(h, batch_vec)  # (BT, embed_dim)
        return emb.view(B, T, self.embed_dim)


    def forward(self, anchors, positives, negatives):
        """Computes embeddings for anchor, positive, and negative samples.

        Args:
            anchors (torch.Tensor): Anchor batch of shape (B, T, V, D).
            positives (torch.Tensor): Positive batch of shape (B, T, V, D).
            negatives (torch.Tensor): Negative batch of shape (B, T, V, D).

        Returns:
            tuple[torch.Tensor, ...]:
                - Anchor embeddings (B, embed_dim)
                - Positive embeddings (B, embed_dim)
                - Negative embeddings (B, embed_dim)
                - Anchor attention weights (B, T, 1)
                - Positive attention weights (B, T, 1)
                - Negative attention weights (B, T, 1)
        """
        a_feats = self.spatial_encode(anchors)
        p_feats = self.spatial_encode(positives)
        n_feats = self.spatial_encode(negatives)
        a_embed, a_attns = self.temporal_pool(a_feats)
        p_embed, p_attns = self.temporal_pool(p_feats)
        n_embed, n_attns = self.temporal_pool(n_feats)

        a_embed = F.normalize(a_embed, p=2, dim=1)
        p_embed = F.normalize(p_embed, p=2, dim=1)
        n_embed = F.normalize(n_embed, p=2, dim=1)
        return a_embed, p_embed, n_embed, a_attns, p_attns, n_attns
    
    def predict(self, x):
        """Predicts embeddings for a batch of sequences.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, V, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Normalized embeddings of shape (B, embed_dim).
                - Attention weights of shape (B, T, 1).
        """
        x = self.spatial_encode(x)
        a_embed, a_attns = self.temporal_pool(x)
        a_embed = F.normalize(a_embed, p=2, dim=1)
        return a_embed, a_attns

