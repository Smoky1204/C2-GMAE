from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from tqdm import trange
from torch_geometric.utils import add_self_loops, remove_self_loops
from utils import set_random_seed, split, load_config, create_optimizer
from evaluation import node_evaluation
from encoder import GraphEncoder
from new_autoencoder import GraphAutoEncoder, compute_density_partition, sym_norm_highpass

def load_torch_data(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)

def train_mae_epoch(graph_auto_encoder, x, edge_index, edge_index_pe, u, PE, xD, optimizer):
    graph_auto_encoder.train()
    loss = graph_auto_encoder(x=x, edge_index=edge_index, u=u, PE=PE, xD=xD, edge_index_pe=edge_index_pe)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def get_split_indices(data, y, seed):
    if 'train_mask' in data.keys():
        if len(data.train_mask.size()) > 1:
            train_idx = torch.where(data.train_mask[:, seed])[0]
            val_idx = torch.where(data.val_mask[:, seed])[0]
            test_idx = torch.where(data.test_mask)[0]
        else:
            train_idx = torch.where(data.train_mask)[0]
            val_idx = torch.where(data.val_mask)[0]
            test_idx = torch.where(data.test_mask)[0]
    else:
        train_idx, val_idx, test_idx = split(y)
    return (train_idx, val_idx, test_idx)

def main():
    parser = ArgumentParser()
    parser.add_argument('--num_exp', type=int, default=5)
    parser.add_argument('--root', type=str, default='./dataset')
    parser.add_argument('--config_dir', type=str, default='./config')
    parser.add_argument('--dataset', type=str, default='blog')
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    config_dir = Path(args.config_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f'{args.dataset}.yaml'
    data_path = root / f'{args.dataset}.pt'
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    if not data_path.exists():
        raise FileNotFoundError(f'Dataset file not found: {data_path}')
    config = load_config(config_path)
    for key, value in config.items():
        setattr(args, key, value)
    args.device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    print(args)
    data = load_torch_data(data_path)
    x = data.x.float().to(args.device)
    edge = data.edge_index.long().to(args.device)
    u = data.u[:, :args.max_freqs].float().to(args.device)
    y = data.y.to(args.device)
    print(y.min().item(), y.max().item())
    edge_index_pe, _ = remove_self_loops(edge, None)
    edge_index_pe, _ = add_self_loops(edge_index_pe, fill_value='mean', num_nodes=u.shape[0])
    PE = torch.linalg.norm(u[edge_index_pe[0]] - u[edge_index_pe[1]], dim=-1)
    xD = sym_norm_highpass(x, edge, x.shape[0])
    log_file_path = log_dir / f'train_log_{args.dataset}.txt'
    final_results = []
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f'Training log for dataset: {args.dataset}\n\n')
        for exp_id in range(args.num_exp):
            args.seed = exp_id
            set_random_seed(exp_id)
            train_idx, val_idx, test_idx = get_split_indices(data, y, args.seed)
            partitions = compute_density_partition(x, edge, num_bins=args.partition_num)
            encoder = GraphEncoder(out_dim=args.embed_dim, args=args).to(args.device)
            model = GraphAutoEncoder(encoder, num_atom_type=args.feat_dim, args=args, partitions=partitions).to(args.device)
            if args.optim != 'sgd':
                args.momentum = None
            optimizer = create_optimizer(opt=args.optim, parameters=model.parameters(), lr=args.init_lr, weight_decay=float(args.weight_decay), momentum=args.momentum)
            if args.use_schedule:
                scheduler_fn = lambda epoch: (1 + np.cos(epoch * np.pi / args.epochs)) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)
            else:
                scheduler = None
            log_file.write(f'=== Experiment {exp_id + 1}/{args.num_exp} ===\n')
            for epoch in trange(1, args.epochs + 1, desc=f'Training Exp {exp_id + 1}/{args.num_exp}'):
                loss = train_mae_epoch(graph_auto_encoder=model, x=x, edge_index=edge, u=u, PE=PE, edge_index_pe=edge_index_pe, xD=xD, optimizer=optimizer)
                if scheduler is not None:
                    scheduler.step()
            model.eval()
            with torch.no_grad():
                embed = model.embed(x, edge, PE)
            acc, pred = node_evaluation(emb=embed, y=y, train_idx=train_idx, valid_idx=val_idx, test_idx=test_idx, epochs=args.epochs_eval, lr=args.lr_eval, weight_decay=args.wd_eval)
            log_file.write(f'Final ACC for Exp {exp_id + 1}: {acc.item():.4f}\n\n')
            print(f'Exp {exp_id + 1}, ACC: {acc.item():.4f}')
            final_results.append(acc.item())
        mean_final_result = np.mean(final_results)
        std_final_result = np.std(final_results)
        print(f'{final_results}')
        print(f'final result: {mean_final_result:.5f}+/-{std_final_result:.5f}')
        log_file.write(f'\nFinal results across {args.num_exp} runs:\n')
        log_file.write(f'Results: {final_results}\n')
        log_file.write(f'Mean+/-Std: {mean_final_result:.5f}+/-{std_final_result:.5f}\n')
if __name__ == '__main__':
    main()
