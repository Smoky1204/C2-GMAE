import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops
from utils import noise_fn, get_activation

class MaskLMHead(nn.Module):

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, mask_tokens=None):
        if mask_tokens is not None:
            features = features[mask_tokens, :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x

class DistanceHead(nn.Module):

    def __init__(self, heads, activation_fn):
        super().__init__()
        self.dense = nn.Linear(heads, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.out_proj = nn.Linear(128, 1)
        self.activation_fn = get_activation(activation_fn)

    def forward(self, dist, edge_index):
        x = self.dense(dist)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        return (x.squeeze(-1), edge_index)

@torch.no_grad()
def _density_cos_mean(x_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    N = x_emb.size(0)
    x = F.normalize(x_emb.float(), p=2, dim=1)
    ei = to_undirected(edge_index)
    ei, _ = add_self_loops(ei, num_nodes=N)
    src, dst = (ei[0], ei[1])
    deg = torch.bincount(dst, minlength=N).clamp_min(1).float()
    nbr_sum = torch.zeros_like(x)
    nbr_sum.index_add_(0, dst, x[src])
    nbr_mean = nbr_sum / deg.unsqueeze(1)
    return (x * nbr_mean).sum(dim=1)

@torch.no_grad()
def _equal_size_partitions_per_graph(batch_vec: torch.Tensor, density: torch.Tensor, K: int) -> torch.Tensor:
    device = density.device
    pid = torch.empty_like(batch_vec, dtype=torch.long, device=device)
    for g in batch_vec.unique():
        idx_g = (batch_vec == g).nonzero(as_tuple=False).view(-1)
        if idx_g.numel() == 0:
            continue
        dens_g = density[idx_g]
        order = torch.argsort(dens_g, descending=True)
        base, rem = (idx_g.numel() // K, idx_g.numel() % K)
        start = 0
        for p in range(K):
            size = base + (1 if p < rem else 0)
            if size == 0:
                continue
            seg_local = order[start:start + size]
            pid[idx_g[seg_local]] = p
            start += size
    return pid

@torch.no_grad()
def _mask_by_partition_per_graph(batch_vec: torch.Tensor, part_id: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    device = batch_vec.device
    chosen_all = []
    for g in batch_vec.unique():
        idx_g = (batch_vec == g).nonzero(as_tuple=False).view(-1)
        if idx_g.numel() == 0:
            continue
        target_M = max(1, int(round(mask_ratio * idx_g.numel())))
        part_g = part_id[idx_g]
        K = int(max(1, part_g.max().item() + 1))
        taken_local = torch.zeros(idx_g.numel(), dtype=torch.bool, device=device)
        chosen_g = []
        for p in range(K):
            idx_gp_local = (part_g == p).nonzero(as_tuple=False).view(-1)
            if idx_gp_local.numel() == 0:
                continue
            t_p = int(round(mask_ratio * idx_gp_local.numel()))
            t = min(t_p, idx_gp_local.numel())
            if t > 0:
                sel_local = idx_gp_local[torch.randperm(idx_gp_local.numel(), device=device)[:t]]
                chosen_g.append(idx_g[sel_local])
                taken_local[sel_local] = True
        if len(chosen_g) == 0:
            chosen_g = [idx_g[:target_M]]
        chosen_g = torch.cat(chosen_g)
        if chosen_g.numel() < target_M:
            rest_local = (~taken_local).nonzero(as_tuple=False).view(-1)
            need = target_M - chosen_g.numel()
            if rest_local.numel() > 0 and need > 0:
                fill = idx_g[rest_local[torch.randperm(rest_local.numel(), device=device)[:need]]]
                chosen_g = torch.cat([chosen_g, fill], dim=0)
        elif chosen_g.numel() > target_M:
            perm = torch.randperm(chosen_g.numel(), device=device)[:target_M]
            chosen_g = chosen_g[perm]
        chosen_all.append(chosen_g)
    return torch.cat(chosen_all) if len(chosen_all) else torch.empty(0, dtype=torch.long, device=device)

@torch.no_grad()
def _hetero_edges_from_feat(x_emb: torch.Tensor, edge_index: torch.Tensor, strategy: str='percentile', top_p: float=0.2, tau: float=0.2) -> torch.Tensor:
    x = F.normalize(x_emb, p=2, dim=-1)
    s, t = (edge_index[0], edge_index[1])
    sim = (x[s] * x[t]).sum(-1)
    if strategy == 'percentile':
        thr = torch.quantile(sim, q=top_p, interpolation='nearest')
        m = sim <= thr
    else:
        m = sim <= tau
    return m.float()

def _sym_highpass(x_emb: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    ei = to_undirected(edge_index)
    ei, _ = add_self_loops(ei, num_nodes=num_nodes)
    src, dst = (ei[1], ei[0])
    deg = torch.bincount(dst, minlength=num_nodes).float()
    dinv = (deg + 1e-12).pow(-0.5)
    w = dinv[dst] * dinv[src]
    Ax = torch.zeros_like(x_emb)
    Ax.index_add_(0, dst, x_emb[src] * w.unsqueeze(-1))
    return x_emb - Ax

class GraphAutoEncoder(nn.Module):

    def __init__(self, encoder, num_atom_type=0, args=None):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.mask_ratio = args.mask_ratio
        self.noise_val = args.noise_val
        self.masked_atom_loss = float(args.masked_atom_loss)
        self.masked_pe_loss = float(args.masked_pe_loss)
        self.atom_recon_type = args.atom_recon_type
        self.num_atom_type = num_atom_type
        self.alpha_l = args.alpha_l
        self.partition_num = getattr(args, 'partition_num', 10)
        self.hetero_strategy = getattr(args, 'hetero_strategy', 'percentile')
        self.hetero_top_p = getattr(args, 'hetero_top_p', 0.2)
        self.hetero_tau = getattr(args, 'hetero_tau', 0.2)
        self.beta_hetero_scale = getattr(args, 'beta_hetero_scale', 0.4)
        self.gamma_hetero_gate = getattr(args, 'gamma_hetero_gate', 1.0)
        self.lambda_H = getattr(args, 'lambda_H', 1.0)
        self.disc_loss_weight = getattr(args, 'disc_loss_weight', 0.2)
        self.kappa_noise_scale = getattr(args, 'kappa_noise_scale', 0.8)
        self.node_pred = MaskLMHead(args.embed_dim, output_dim=self.num_atom_type, activation_fn=args.task_head_activation)
        self.pe_reconstruct_heads = DistanceHead(heads=args.embed_dim, activation_fn=args.task_head_activation)
        self.disc_head = MaskLMHead(args.embed_dim, output_dim=args.embed_dim, activation_fn=args.task_head_activation)

    def forward(self, batch):
        x, edge_index, edge_attr = (batch.x, batch.edge_index, batch.edge_attr)
        snorm_n = batch.snorm_n
        batch_vec = batch.batch
        e, u = (batch.EigVals.clone(), batch.EigVecs.clone())
        mask_u = torch.isnan(u)
        u[mask_u] = 0
        mask_e = torch.isnan(e)
        e[mask_e] = 0
        with torch.no_grad():
            x_emb_for_density = self.encoder.x_embedding(x)
        density = _density_cos_mean(x_emb_for_density, edge_index)
        part_id = _equal_size_partitions_per_graph(batch_vec, density, self.partition_num)
        mask_nodes = _mask_by_partition_per_graph(batch_vec, part_id, self.mask_ratio)
        out_x = x.clone()
        if mask_nodes.numel() > 0:
            out_x[mask_nodes, 0] = self.num_atom_type
        with torch.no_grad():
            x_emb_for_hetero = F.normalize(self.encoder.x_embedding(x), p=2, dim=-1)
        M = _hetero_edges_from_feat(x_emb_for_hetero, edge_index, strategy=self.hetero_strategy, top_p=self.hetero_top_p, tau=self.hetero_tau)
        u_masked = u.clone()
        if mask_nodes.numel() > 0:
            pos_noise = noise_fn(self.noise_val, len(mask_nodes), u.size(1)).to(u_masked.device)
            ei_ud = to_undirected(edge_index)
            N = x.size(0)
            one = torch.ones(ei_ud.size(1), device=x.device)
            degT = torch.zeros(N, device=x.device)
            degT.index_add_(0, ei_ud[0], one)
            degT.index_add_(0, ei_ud[1], one)
            degH = torch.zeros(N, device=x.device)
            degH.index_add_(0, ei_ud[0], M)
            degH.index_add_(0, ei_ud[1], M)
            ratio = degH / (degT + 1e-08)
            scale = (1.0 + self.kappa_noise_scale * ratio[mask_nodes]).unsqueeze(1)
            pos_noise = pos_noise * scale
            u_masked[mask_nodes] += pos_noise
        PE = torch.linalg.norm(u[edge_index[0]] - u[edge_index[1]], dim=-1)
        PE_noise = torch.linalg.norm(u_masked[edge_index[0]] - u_masked[edge_index[1]], dim=-1)
        PE_mod = PE * (1.0 + self.beta_hetero_scale * M)
        PE_noise_mod = PE_noise * (1.0 + self.beta_hetero_scale * M)
        pe_bias = self.gamma_hetero_gate * M
        enc_rep, pe = self.encoder(out_x, out_x, edge_index, edge_attr=edge_attr, snorm_n=snorm_n, PE=PE_mod, PE_noise=PE_noise_mod, pe_bias=pe_bias)
        pred_masked_feat = self.node_pred(enc_rep, mask_tokens=mask_nodes)
        atom_loss = self.cal_atom_loss(pred_node=pred_masked_feat, target_atom=x, mask_tokens=mask_nodes, loss_fn=self.atom_recon_type, alpha_l=self.alpha_l)
        reconstruct_dist, ei_used = self.pe_reconstruct_heads(pe, edge_index)
        pe_loss = self.cal_pe_loss(reconstruct_dis=reconstruct_dist, target_dis=PE_mod, edge_index=ei_used, mask_tokens=mask_nodes, hetero_mask=M, lambda_H=self.lambda_H)
        x_sem = self.encoder.x_embedding(x)
        x_high = _sym_highpass(x_sem, edge_index, x.size(0))
        unmask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        if mask_nodes.numel():
            unmask[mask_nodes] = False
        pred_disc = self.disc_head(enc_rep, mask_tokens=None)
        disc_loss = self.sce_loss(pred_disc[unmask], x_high[unmask], alpha=self.alpha_l)
        loss = self.masked_atom_loss * atom_loss + self.masked_pe_loss * pe_loss + self.disc_loss_weight * disc_loss
        return loss

    def embed(self, batch):
        x, edge_index, edge_attr = (batch.x, batch.edge_index, batch.edge_attr)
        snorm_n = batch.snorm_n
        e, u = (batch.EigVals.clone(), batch.EigVecs.clone())
        mask_u = torch.isnan(u)
        u[mask_u] = 0
        mask_e = torch.isnan(e)
        e[mask_e] = 0
        PE = torch.linalg.norm(u[edge_index[0]] - u[edge_index[1]], dim=-1)
        enc_rep, _ = self.encoder.embed(x, edge_index, edge_attr=edge_attr, snorm_n=snorm_n, PE=PE, pe_bias=None)
        return enc_rep

    def cal_pe_loss(self, reconstruct_dis, target_dis, edge_index, mask_tokens, hetero_mask, lambda_H=1.0):
        row = edge_index[0]
        idx = torch.isin(row, mask_tokens)
        if not idx.any():
            return torch.tensor(0.0, device=target_dis.device)
        base = F.smooth_l1_loss(reconstruct_dis[idx], target_dis[idx], reduction='none', beta=1.0)
        w = 1.0 + lambda_H * hetero_mask[idx]
        return (base * w).mean()

    def cal_atom_loss(self, pred_node, target_atom, mask_tokens, loss_fn, alpha_l=0.0):
        if mask_tokens.numel() == 0:
            return torch.tensor(0.0, device=pred_node.device)
        target_cls = target_atom[mask_tokens, 0]
        if loss_fn == 'sce':
            target = F.one_hot(target_cls, num_classes=self.num_atom_type).float()
            return self.sce_loss(pred_node, target, alpha=alpha_l)
        elif loss_fn == 'mse':
            target = F.one_hot(target_cls, num_classes=self.num_atom_type).float()
            return self.mse_loss(pred_node, target)
        else:
            return nn.CrossEntropyLoss()(pred_node, target_cls)

    @staticmethod
    def sce_loss(x, y, alpha=1):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return (1 - (x * y).sum(dim=-1)).pow_(alpha).mean()

    @staticmethod
    def mse_loss(x, y):
        return ((x - y) ** 2).mean()
