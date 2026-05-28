from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import remove_self_loops, to_undirected, add_self_loops
from utils import get_activation, noise_fn


def compute_node_density(x, edge_index):

    x_norm = F.normalize(x)
    N = x.size(0)
    from torch_geometric.utils import to_scipy_sparse_matrix
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=N).tocsr()

    density = torch.zeros(N, device=x.device)
    for i in range(N):
        first_neighbors = set(adj[i].indices)
        second_neighbors = set()
        for n in first_neighbors:
            second_neighbors.update(adj[n].indices)
        all_neighbors = first_neighbors.union(second_neighbors)
        all_neighbors.discard(i)
        if len(all_neighbors) > 0:
            all_neighbors = list(all_neighbors)
            sim = torch.matmul(x_norm[i], x_norm[all_neighbors].T)
            density[i] = sim.mean()
        else:
            density[i] = 0.0
    return density


def compute_density_partition(x, edge_index, num_bins=4):
    density = compute_node_density(x, edge_index)
    sorted_indices = torch.argsort(density, descending=True)
    num_nodes = x.size(0)
    bin_size = num_nodes // num_bins
    partitions = []
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < num_bins - 1 else num_nodes
        partitions.append(sorted_indices[start:end])
    return partitions


def sample_mask_nodes(partitions, mask_ratio, num_nodes, device):
    if not partitions:
        return torch.tensor([], dtype=torch.long, device=device)
    num_mask_nodes = int(mask_ratio * num_nodes)
    nodes_per_partition = max(1, num_mask_nodes // max(1, len(partitions)))
    mask_nodes = []
    for part in partitions:
        if len(part) > 0:
            num_to_sample = min(nodes_per_partition, len(part))
            rand_idx = torch.randperm(len(part), device=device)[:num_to_sample]
            mask_nodes.append(part[rand_idx])
    if not mask_nodes:
        return torch.tensor([], dtype=torch.long, device=device)
    return torch.cat(mask_nodes).to(device)


class DistanceHead(nn.Module):
    def __init__(self, heads, activation_fn):
        super().__init__()
        self.dense = nn.Linear(heads, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.out_proj = nn.Linear(128, 1)
        self.activation_fn = get_activation(activation_fn)

    def forward(self, dist, edge_index_pe):
        from torch_geometric.utils import remove_self_loops, to_undirected
        edge_index, dist = remove_self_loops(edge_index=edge_index_pe, edge_attr=dist)
        dist = self.dense(dist)
        dist = self.activation_fn(dist)
        dist = self.layer_norm(dist)
        dist = self.out_proj(dist)  
        edge_index, dist = to_undirected(edge_index=edge_index, edge_attr=dist, reduce="mean")
        return dist.squeeze(), edge_index


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


@torch.no_grad()
def compute_hetero_mask_by_feature(x, edge_index_pe, strategy="percentile", top_p=0.2, tau=0.2):

    x_n = F.normalize(x, p=2, dim=-1)
    src, dst = edge_index_pe[0], edge_index_pe[1]
    sim = (x_n[src] * x_n[dst]).sum(dim=-1)
    if strategy == "percentile":
        thr = torch.quantile(sim, q=top_p, interpolation="nearest")
        het = (sim <= thr)
    else:
        het = (sim <= tau)
    return het.float()


def compute_degrees(edge_index, num_nodes):
    deg = torch.bincount(edge_index[0], minlength=num_nodes) 
    deg = deg + torch.bincount(edge_index[1], minlength=num_nodes) 
    return deg


def compute_hetero_degree(mask_edge_float, edge_index_pe, num_nodes):

    ei, m = remove_self_loops(edge_index_pe, mask_edge_float)
    ei, m = to_undirected(ei, m, reduce="mean")
    src, dst = ei[0], ei[1]
    deg_total = compute_degrees(ei, num_nodes).to(mask_edge_float.device).float()
    degH = torch.zeros(num_nodes, device=mask_edge_float.device)
    degH.index_add_(0, src, m)
    degH.index_add_(0, dst, m)
    return degH, deg_total, ei, m  


def sym_norm_highpass(x, edge_index, num_nodes):

    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index_sl
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    adj = torch.sparse_coo_tensor(
        edge_index_sl, 
        edge_weight, 
        (num_nodes, num_nodes)
    ).to(x.device) 

    Ax = torch.sparse.mm(adj, x)
    xD = x - Ax
    return xD



class GraphAutoEncoder(nn.Module):
    def __init__(self, encoder, num_atom_type=0, args=None, partitions=None):
        super(GraphAutoEncoder, self).__init__()
        self.args = args
        self.encoder = encoder

        self.mask_ratio = args.mask_ratio
        self.replace_ratio = args.replace_ratio
        self.noise_val = args.noise_val
        self.masked_atom_loss = float(args.masked_atom_loss)
        self.masked_pe_loss = float(args.masked_pe_loss)
        self.atom_recon_type = args.atom_recon_type
        self.num_atom_type = num_atom_type
        self.alpha_l = args.alpha_l


        self.beta_hetero_scale = getattr(args, 'beta_hetero_scale', 0.4)   
        self.gamma_hetero_gate = getattr(args, 'gamma_hetero_gate', 1.0)   
        self.kappa_noise_scale = getattr(args, 'kappa_noise_scale', 0.8)   
        self.lambda_H = getattr(args, 'lambda_H', 1.0)                     
        self.disc_loss_weight = getattr(args, 'disc_loss_weight', 0.2)     
        self.hetero_strategy = getattr(args, 'hetero_strategy', 'percentile')
        self.hetero_top_p = getattr(args, 'hetero_top_p', 0.2)
        self.hetero_tau = getattr(args, 'hetero_tau', 0.2)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, args.feat_dim))
        self.node_pred = MaskLMHead(args.embed_dim, output_dim=self.num_atom_type, activation_fn=args.task_head_activation)
        self.pe_reconstruct_heads = DistanceHead(heads=args.heads, activation_fn=args.task_head_activation)
        self.disc_head = MaskLMHead(args.embed_dim, output_dim=self.num_atom_type, activation_fn=args.task_head_activation)

        self.partitions = partitions  

    def forward(self, x, edge_index, u, PE, xD, edge_index_pe=None):
        N = x.size(0)
        device = x.device

        hetero_edge_mask = compute_hetero_mask_by_feature(
            x, edge_index_pe, strategy=self.hetero_strategy, top_p=self.hetero_top_p, tau=self.hetero_tau
        ) 
        degH, degT, ei_posloss, hetero_mask_u = compute_hetero_degree(hetero_edge_mask, edge_index_pe, N)
        x_masked, u_masked, mask_tokens = self.encoding_mask_noise(
            x=x, u=u, degH=degH, degT=degT
        )
        PE_noise = torch.linalg.norm(u_masked[edge_index_pe[0]] - u_masked[edge_index_pe[1]], dim=-1)
        M = hetero_edge_mask
        PE_mod = PE * (1.0 + self.beta_hetero_scale * M)
        PE_noise_mod = PE_noise * (1.0 + self.beta_hetero_scale * M)
        pe_bias = self.gamma_hetero_gate * M
        h_latent, pe_encoded = self.encoder(x, x_masked, edge_index, PE=PE_mod, PE_noise=PE_noise_mod, pe_bias=pe_bias)
        pred_masked_feat = self.node_pred(h_latent, mask_tokens)
        
        reconstruct_dist, undirected_ei = self.pe_reconstruct_heads(pe_encoded, edge_index_pe)
        atom_loss = self.cal_atom_loss(pred_node=pred_masked_feat, target_atom=x, mask_tokens=mask_tokens,
                                       loss_fn=self.atom_recon_type, alpha_l=self.alpha_l)
        pe_loss = self.cal_pe_loss(reconstruct_dis=reconstruct_dist, target_dis=PE_mod,
                                   edge_index_pe=edge_index_pe, mask_tokens=mask_tokens,
                                   hetero_mask=hetero_edge_mask, lambda_H=self.lambda_H)

        unmask = torch.ones(N, dtype=torch.bool, device=device)
        unmask[mask_tokens] = False
        pred_disc_all = self.disc_head(h_latent, mask_tokens=None)
        disc_loss = self.sce_loss(pred_disc_all[unmask], xD[unmask], alpha=self.alpha_l)

        loss = self.masked_atom_loss * atom_loss + self.masked_pe_loss * pe_loss + self.disc_loss_weight * disc_loss
        return loss

    def encoding_mask_noise(self, x, u, degH=None, degT=None):
        mask_token_ratio = 1 - self.replace_ratio
        num_nodes = x.size(0)
        mask_nodes = sample_mask_nodes(self.partitions, self.mask_ratio, num_nodes, x.device)

        out_x = x.clone()
        if self.replace_ratio > 0 and mask_nodes.numel() > 0:
            num_mask_nodes = mask_nodes.size(0)
            num_noise_nodes = int(self.replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(mask_token_ratio * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            token_nodes = mask_nodes
            if token_nodes.numel() > 0:
                out_x[token_nodes] = 0.0

        if token_nodes.numel() > 0:
            out_x[token_nodes] += self.enc_mask_token

        u_masked = u.clone()
        pos_noise = noise_fn(self.noise_val, len(mask_nodes), u.size(1)).to(u.device) if mask_nodes.numel() > 0 else torch.empty(0, u.size(1), device=u.device)
        if mask_nodes.numel() > 0:
            if degH is not None and degT is not None:
                ratio = degH / (degT + 1e-8)    
                scale = (1.0 + self.kappa_noise_scale * ratio[mask_nodes]).unsqueeze(1)  
                pos_noise = pos_noise * scale
            u_masked[mask_nodes] += pos_noise

        return out_x, u_masked, mask_nodes

    def embed(self, x, edge_index, PE):
        h, _ = self.encoder.embed(x, edge_index, PE=PE, pe_bias=None)
        return h

    def cal_pe_loss(self, reconstruct_dis, target_dis, edge_index_pe, mask_tokens, hetero_mask, lambda_H=1.0):
        edge_index_rm, target_dis_rm = remove_self_loops(edge_index=edge_index_pe, edge_attr=target_dis)
        edge_index_ud, target_dis_ud = to_undirected(edge_index=edge_index_rm, edge_attr=target_dis_rm, reduce="mean")
        edge_index_rm2, hetero_mask_rm = remove_self_loops(edge_index=edge_index_pe, edge_attr=hetero_mask.float())
        edge_index_ud2, hetero_mask_ud = to_undirected(edge_index=edge_index_rm2, edge_attr=hetero_mask_rm, reduce="mean")
        assert edge_index_ud.size(1) == edge_index_ud2.size(1), "Edge alignment mismatch in PE loss"

        row = edge_index_ud[0]
        idx = torch.isin(row, mask_tokens)
        base = F.smooth_l1_loss(reconstruct_dis[idx], target_dis_ud[idx], reduction="none", beta=1.0)
        weight = 1.0 + lambda_H * hetero_mask_ud[idx]
        return (base * weight).mean()

    def cal_atom_loss(self, pred_node, target_atom, mask_tokens, loss_fn, alpha_l=0.0):
        target_atom = target_atom[mask_tokens]
        if target_atom.numel() == 0:
            return torch.tensor(0.0, device=pred_node.device)
        if loss_fn == "sce":
            return self.sce_loss(pred_node, target_atom, alpha=alpha_l)
        elif loss_fn == "mse":
            return self.mse_loss(pred_node, target_atom)
        else:
            return nn.CrossEntropyLoss()(pred_node, target_atom)

    def sce_loss(self, x, y, alpha=1):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return ((1 - (x * y).sum(dim=-1)).pow(alpha)).mean()

    def mse_loss(self, x, y):
        return ((x - y) ** 2).mean()