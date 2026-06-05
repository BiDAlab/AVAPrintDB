import torch
import torch.nn as nn
from tqdm import tqdm
import sys

import torch.nn.functional as F
from abc import ABC, abstractmethod
import torch.nn as nn
from plotting import plot_score_histograms

class BaseModel(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def train_one_epoch(self, dataloader, optimizer, loss_fn, **kwargs):
        pass

    @abstractmethod
    def validate(self, dataloader, loss_fn, **kwargs):
        pass

    @abstractmethod
    def predict(self, input_data, **kwargs):
        pass


class ContentTemporalAttention(nn.Module):
    """
    Content-based multi-head temporal attention pooling.
    Input:  (B, T, D)
    Output: (B, D)
    """
    def __init__(self, embedding_dim=768, num_heads=4, dropout=0.0, max_len=512, pos_dropout=0.0):
        super().__init__()

        project_dim_attn = 64 
        self.norm = nn.LayerNorm(embedding_dim)
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(pos_dropout) if pos_dropout > 0 else nn.Identity()

        self.score_mlp = nn.Sequential(
            nn.Linear(embedding_dim, project_dim_attn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(project_dim_attn, num_heads)
        )

    def forward(self, x, mask=None, return_attn=False):
        B, T, D = x.shape

        x = self.pos_drop(x + self.pos_embed[:, :T, :])
        x = self.norm(x)

        scores = self.score_mlp(x)  # (B, T, H)

        if mask is not None:
            scores = scores.masked_fill(mask[:, :, None] == 0, -1e9)

        attn = F.softmax(scores, dim=1)      # temporal softmax over T
        pooled = torch.einsum("btd,bth->bhd", x, attn)  # (B, H, D)
        pooled = pooled.mean(dim=1)          # (B, D)

        if return_attn:
            return pooled, attn, scores
        return pooled

class TemporalAttentionPoolingModelNew(BaseModel):


    def __init__(
        self,
        backbone=None,
        embedding_dim=768,     # DINO/CLIP feature dim (512/768/1024...)
        project_dim=256,       # hidden dim inside attention scorer AND projection MLP
        output_dim=256,        # final embedding dim
        num_heads=4,
        dropout=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.margin = kwargs.get("margin", 0.3)

        # Temporal pooling (attention)
        self.temporal_pooling = ContentTemporalAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.2
        )

        # Projection head: D -> project_dim -> output_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, project_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(project_dim, output_dim),
        )

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        self.output_dir = kwargs.get("save_model_to_dir", './outputHist')
        self.output_dir_plots = kwargs.get("save_model_to_dir", './outputPlots') + "/plots"
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir_plots, exist_ok=True)

    def get_embedding(self, feats, mask=None, return_attn=False, **kwargs):
        if return_attn:
            pooled, attn, scores = self.temporal_pooling(feats, mask=mask, return_attn=return_attn)
            embedding = self.projection(pooled)
            return embedding, attn, scores
        
        pooled = self.temporal_pooling(feats, mask=mask)     # (B, D)
        embedding = self.projection(pooled)                  # (B, output_dim)
        return embedding
    
    def run_backbone(self, videos):
        BS, T, C, H, W = videos.shape
        x = videos.view(BS * T, C, H, W)
    
        with torch.no_grad():
            feats = self.backbone(x)  # [BS*T, D]
        return feats.view(BS, T, -1)


    @staticmethod
    def make_mask_from_tensor(videos):
        with torch.no_grad():
            frame_sums = videos.abs().sum(dim=(2))  # shape: (B*3, T)
            mask = frame_sums > 0  # (B*3, T), boolean
        return mask
    


    def forward(self, videos, labels):
        # videos: Tensor of shape [B*3, T_max, D]
        videos = videos.to(torch.float32)
        Total, B, T, D = videos.shape

        videos_flat = videos.flatten(0,1)
        embeddings_flat = self.get_embedding(videos_flat)
        
        anch_labels = (labels == 0)
        pos_labels = (labels == 1)
        neg_labels = (labels == -1)

        embeddings = embeddings_flat.view(Total, B, -1)  # (Total, B, D)
        _,_, D_emb = embeddings.shape

        a_embeds = embeddings[anch_labels].view(Total, -1, D_emb)    
        p_embeds = embeddings[pos_labels].view(Total, -1, D_emb)           
        n_embeds = embeddings[neg_labels].view(Total, -1, D_emb)  

        return a_embeds, p_embeds, n_embeds


    def train_one_epoch(self, dataloader, optimizer, loss_fn=None, **kwargs):
    
        self.train()
        device = next(self.parameters()).device
        
        total_loss = 0.0
        num_batches = 0.0

        pbar = tqdm(enumerate(dataloader), file=sys.stdout, desc="Processing batches", total=len(dataloader))

        y_scores, y_true = [], []  # For storing distances and labels for evaluation

        for i,batch in pbar:
            optimizer.zero_grad()
            batch, labels = batch
            batch = batch.to(device, non_blocking=True)

            # Obtener embeddings triplet
            anchor, positive, negative = self.forward(batch, labels)  # Shapes: (N, B, D)

            # Normalizamos en la última dimensión (la del embedding)
            a_embeds = F.normalize(anchor, p=2, dim=-1, eps=1e-6)
            p_embeds = F.normalize(positive, p=2, dim=-1, eps=1e-6)
            n_embeds = F.normalize(negative, p=2, dim=-1, eps=1e-6)

            dists_p = torch.norm(p_embeds - a_embeds, p=2, dim=-1).T
            dists_n = torch.norm(n_embeds - a_embeds, p=2, dim=-1).T

            loss = self.lossCalculate(dists_p, dists_n)

            d_p = dists_p.reshape(-1)
            d_n = dists_n.reshape(-1)

            l_p = torch.zeros_like(d_p)  # Positive pairs should have distance close to 0
            l_n = torch.ones_like(d_n)  # Negative pairs should have distance larger

            y_scores.append(torch.cat([d_p, d_n], dim=0).detach().cpu())
            y_true.append(torch.cat([l_p, l_n], dim=0).detach().cpu())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if i % 10 == 0:
                pbar.set_postfix({"batch_loss": loss.item(), "train_loss": total_loss / num_batches})

        avg_loss = total_loss / num_batches
    
        pbar.set_postfix({"train_loss": avg_loss})
        # return total_loss, num_batches

        y_scores = torch.cat(y_scores, dim=0).detach().cpu().numpy()
        y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()

        plot_score_histograms(y_scores, y_true, score_histograms_path=f"{self.output_dir_plots}/train_score_histogram.png")
        
        avg_loss = total_loss / num_batches
    
        pbar.set_postfix({"train_loss": avg_loss})
        return total_loss, num_batches

    def validate(self, dataloader, loss_fn=None, **kwargs):
    
        self.eval()
        device = next(self.parameters()).device

        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(enumerate(dataloader), file=sys.stdout, desc="Processing validation batches", total=len(dataloader))

        y_scores, y_true = [], []

        with torch.no_grad():
            for i, batch in pbar:

                batch, labels = batch

                # batch = torch.stack(batch, dim=1).to(device)
                batch = batch.to(device, non_blocking=True)
                anchor, positive, negative = self.forward(batch, labels) 

                # Normalizamos en la última dimensión (la del embedding)
                a_embeds = F.normalize(anchor, p=2, dim=-1)
                p_embeds = F.normalize(positive, p=2, dim=-1)
                n_embeds = F.normalize(negative, p=2, dim=-1)

                dists_p = torch.norm(p_embeds - a_embeds, p=2, dim=-1).T
                dists_n = torch.norm(n_embeds - a_embeds, p=2, dim=-1).T

                loss = self.lossCalculate(dists_p, dists_n)

                d_p = dists_p.reshape(-1)
                d_n = dists_n.reshape(-1)

                l_p = torch.zeros_like(d_p)  # Positive pairs should have distance close to 0
                l_n = torch.ones_like(d_n)  # Negative pairs should have distance larger

                y_scores.append(torch.cat([d_p, d_n], dim=0).detach().cpu())
                y_true.append(torch.cat([l_p, l_n], dim=0).detach().cpu())

                total_loss += loss.item()
                num_batches += 1

                if i % 10 == 0:
                    # Update progress bar with current batch loss and average loss
                    pbar.set_postfix({"batch_loss":loss.item(), "val_loss": total_loss /num_batches})

            y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
            y_true = torch.cat(y_true, dim=0).cpu().numpy()

            plot_score_histograms(y_scores, y_true, score_histograms_path=f"{self.output_dir_plots}/val_score_histogram.png")

        avg_loss = total_loss / num_batches
    
        return total_loss, num_batches

    def lossCalculate(self, dists_p, dists_n):
        all_combinations = dists_p.unsqueeze(1) - dists_n.unsqueeze(0) # [12, 12, 32]

        loss_semi = torch.relu(all_combinations + self.margin).mean()

        hard_p = dists_p.max(dim=0)[0]
        hard_n = dists_n.min(dim=0)[0]

        loss_hard = torch.relu(hard_p - hard_n + self.margin).mean()

        alpha = 0.7
        loss = alpha * loss_semi + (1 - alpha) * loss_hard
        return loss

    def predict(self, input_data, return_attn=False, **kwargs):
        self.eval()
        with torch.no_grad():
            feats = input_data.to(torch.float32)
            if return_attn:
                embedding, attn, scores = self.get_embedding(feats, return_attn=return_attn)
                return embedding, attn, scores
            feats = self.get_embedding(feats)
        return feats
    



