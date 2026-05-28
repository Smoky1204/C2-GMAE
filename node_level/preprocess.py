import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_undirected
SUPPORTED_DATASETS = {'blog', 'actor', 'chameleon', 'squirrel', 'facebook', 'wiki'}

def _as_path(path):
    return Path(path).expanduser().resolve()

def _get_npz_array(data_dict, candidates):
    for key in candidates:
        if key in data_dict:
            return data_dict[key]
    raise KeyError(f'None of these keys were found in the npz file: {candidates}')

def _fix_mask_shape(mask, num_nodes):
    mask = np.asarray(mask)
    if mask.ndim == 2 and mask.shape[0] != num_nodes and (mask.shape[1] == num_nodes):
        mask = mask.T
    return torch.from_numpy(mask).bool()

def load_blog(root):
    dataset_dir = root / 'blog'
    adj_path = dataset_dir / 'adj.npz'
    feat_path = dataset_dir / 'feat.npz'
    label_path = dataset_dir / 'label.npy'
    train_path = dataset_dir / 'train20.npy'
    val_path = dataset_dir / 'val.npy'
    test_path = dataset_dir / 'test.npy'
    required_files = [adj_path, feat_path, label_path, train_path, val_path, test_path]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing BlogCatalog files:\n' + '\n'.join(missing))
    adj = sps.load_npz(adj_path).tocoo()
    features = sps.load_npz(feat_path)
    labels = np.load(label_path)
    idx_train = np.load(train_path)
    idx_val = np.load(val_path)
    idx_test = np.load(test_path)
    edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
    x = torch.tensor(features.toarray(), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = x.size(0)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[idx_train] = True
    data.val_mask[idx_val] = True
    data.test_mask[idx_test] = True
    return data

def load_manual_npz(dataset_name, root):
    if dataset_name == 'facebook':
        npz_path = root / 'FacebookPagePage' / 'raw' / 'facebook.npz'
        norm_type = 'l1'
    elif dataset_name == 'wiki':
        npz_path = root / 'WikiCS' / 'raw' / 'wiki.npz'
        norm_type = 'l2'
    else:
        raise ValueError(f'Unsupported manual npz dataset: {dataset_name}')
    if not npz_path.exists():
        raise FileNotFoundError(f'Cannot find npz file: {npz_path}')
    data_dict = np.load(npz_path, allow_pickle=True)
    features = _get_npz_array(data_dict, ['features', 'x', 'feat'])
    labels = _get_npz_array(data_dict, ['target', 'labels', 'y'])
    edges = _get_npz_array(data_dict, ['edges', 'edge_index'])
    x = torch.from_numpy(features).float()
    y = torch.from_numpy(labels).long()
    edge_index = torch.from_numpy(edges).long()
    if edge_index.dim() != 2:
        raise ValueError(f'edge_index must be a 2D array, got shape {tuple(edge_index.shape)}')
    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.size(0) != 2:
        raise ValueError(f'edge_index must have shape [2, E], got {tuple(edge_index.shape)}')
    if norm_type == 'l1':
        row_sum = x.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        x = x / row_sum
    elif norm_type == 'l2':
        x = F.normalize(x, p=2, dim=1)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = x.size(0)
    if dataset_name == 'wiki':
        if 'train_masks' in data_dict:
            data.train_mask = _fix_mask_shape(data_dict['train_masks'], data.num_nodes)
        if 'val_masks' in data_dict:
            data.val_mask = _fix_mask_shape(data_dict['val_masks'], data.num_nodes)
        if 'stopping_masks' in data_dict:
            data.stopping_mask = _fix_mask_shape(data_dict['stopping_masks'], data.num_nodes)
        if 'test_mask' in data_dict:
            data.test_mask = _fix_mask_shape(data_dict['test_mask'], data.num_nodes)
    return data

def load_pyg_dataset(dataset_name, root):
    if dataset_name == 'actor':
        dataset = Actor(root=str(root / 'Actor'), transform=NormalizeFeatures())
    elif dataset_name in {'chameleon', 'squirrel'}:
        dataset = WikipediaNetwork(root=str(root / 'WikipediaNetwork'), name=dataset_name, transform=NormalizeFeatures())
    else:
        raise ValueError(f'Unsupported PyG dataset: {dataset_name}')
    return dataset[0]

def load_dataset(dataset_name, root):
    dataset_name = dataset_name.lower()
    if dataset_name == 'blog':
        data = load_blog(root)
    elif dataset_name in {'facebook', 'wiki'}:
        data = load_manual_npz(dataset_name, root)
    elif dataset_name in {'actor', 'chameleon', 'squirrel'}:
        data = load_pyg_dataset(dataset_name, root)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    data.name = dataset_name
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return data

def compute_laplacian_eigs(data, k_eigs=50, full_eigs=False, tol=1e-05):
    num_nodes = data.num_nodes
    index, attr = get_laplacian(data.edge_index, normalization='sym', num_nodes=num_nodes)
    laplacian = to_scipy_sparse_matrix(index, attr, num_nodes=num_nodes).asfptype()
    if full_eigs or k_eigs >= num_nodes - 1:
        dense_laplacian = torch.from_numpy(laplacian.toarray()).float()
        eigvals, eigvecs = torch.linalg.eigh(dense_laplacian)
        if not full_eigs:
            eigvals = eigvals[:k_eigs]
            eigvecs = eigvecs[:, :k_eigs]
        return (eigvals.float(), eigvecs.float())
    k = max(1, min(k_eigs, num_nodes - 2))
    try:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(laplacian, k=k, which='SM', tol=tol)
    except Exception:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(laplacian, k=k, which='SM', tol=max(tol, 0.0001))
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return (torch.from_numpy(eigvals).float(), torch.from_numpy(eigvecs).float())

def save_processed_data(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess node classification datasets for C2-GMAE.')
    parser.add_argument('--dataset', type=str, required=True, choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument('--root', type=str, default='./dataset')
    parser.add_argument('--k_eigs', type=int, default=50)
    parser.add_argument('--full_eigs', action='store_true')
    parser.add_argument('--tol', type=float, default=1e-05)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    root = _as_path(args.root)
    output_path = _as_path(args.output) if args.output else root / f'{args.dataset}.pt'
    print(f'Loading dataset: {args.dataset}')
    data = load_dataset(args.dataset, root)
    print(f'Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}, Feature dim: {data.x.size(1)}')
    print(f'Computing Laplacian eigenvectors with k={args.k_eigs}')
    eigvals, eigvecs = compute_laplacian_eigs(data=data, k_eigs=args.k_eigs, full_eigs=args.full_eigs, tol=args.tol)
    data.e = eigvals
    data.u = eigvecs
    save_processed_data(data, output_path)
    print(f'Saved processed data to: {output_path}')
if __name__ == '__main__':
    main()
