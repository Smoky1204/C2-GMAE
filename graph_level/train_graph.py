import argparse
import os
import time
import datetime
import numpy as np
import torch
from torch import optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from ogb.graphproppred import Evaluator
from tqdm import trange
from dataset_ogb import OGBDataset
from dataloader import collate_fn
from autoencoder import GraphAutoEncoder
from encoder import GraphEncoder
from evaluation import ogbg_evaluation
from utils import load_config, create_optimizer, create_schedule, set_random_seed

def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:02d}'

def train_mae_epoch(graph_auto_encoder, data_loader, optimizer, device):
    graph_auto_encoder.train()
    t0 = time.time()
    total_loss = 0.0
    steps = 0
    for step, batch in enumerate(data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = graph_auto_encoder(batch)
        loss.backward()
        optimizer.step()
        steps += 1
        total_loss += float(loss.detach().item())
    epoch_sec = time.time() - t0
    avg_loss = total_loss / max(steps, 1)
    return (avg_loss, steps, epoch_sec)

def evaluation_network(model, data_loader, pooling, device):
    model.eval()
    emb = []
    y_true = []
    for batch in data_loader:
        with torch.no_grad():
            batch = batch.to(device)
            out = model.embed(batch)
            if pooling == 'mean':
                out = global_mean_pool(out, batch.batch)
            elif pooling == 'max':
                out = global_max_pool(out, batch.batch)
            elif pooling == 'sum':
                out = global_add_pool(out, batch.batch)
            else:
                raise NotImplementedError
        emb.append(out.detach().cpu())
        y_true.append(batch.y.cpu())
    emb = torch.cat(emb, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()
    return (emb, y_true)

def main(args):
    per_round_train = []
    per_round_valid = []
    per_round_test = []
    final_results = []
    for ep_num in range(args.num_exp):
        args.seed = ep_num
        set_random_seed(ep_num)
        torch.cuda.manual_seed_all(ep_num)
        processed_name = f'processed_{args.lap_norm}'
        dataset = OGBDataset(root=args.root, dataset=args.dataset, max_freqs=args.max_freqs, lap_norm=args.lap_norm, processed_name=processed_name)
        split_idx = dataset.get_idx_split()
        task_type = dataset.task_type
        args.num_tasks = dataset.num_tasks
        train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)
        evaluator = Evaluator(name=args.dataset)
        encoder = GraphEncoder(out_dim=args.num_atom_type, args=args).to(args.device)
        model = GraphAutoEncoder(encoder=encoder, num_atom_type=args.num_atom_type, args=args).to(args.device)
        parameters = model.parameters()
        if args.optim == 'sgd':
            momentum = args.momentum
        else:
            momentum = None
        optimizer = create_optimizer(opt=args.optim, parameters=parameters, lr=args.init_lr, weight_decay=float(args.weight_decay), momentum=momentum)
        if args.use_scheduler:
            scheduler = create_schedule(opt=args.schedule_opt, optimizer=optimizer, max_epoch=args.epochs, num_warmup_steps=args.num_warmup_epochs)
        else:
            scheduler = None
        print('=' * 80)
        print(f'[Exp {ep_num + 1}/{args.num_exp}] Dataset={args.dataset}  Epochs={args.epochs}  BatchSize={args.batch_size}  Device={args.device}')
        try:
            ds_len = len(train_loader.dataset)
            steps_per_epoch = len(train_loader)
            print(f'Training samples: {ds_len} | Steps/epoch: {steps_per_epoch}')
        except Exception:
            pass
        print('=' * 80)
        global_t0 = time.time()
        epoch_durations = []
        pbar = trange(1, args.epochs + 1, desc=f'Exp {ep_num + 1}/{args.num_exp}', dynamic_ncols=True, leave=True)
        for epoch in pbar:
            avg_loss, steps, epoch_sec = train_mae_epoch(graph_auto_encoder=model, data_loader=train_loader, optimizer=optimizer, device=args.device)
            if scheduler:
                scheduler.step()
            epoch_durations.append(epoch_sec)
            avg_epoch = float(np.mean(epoch_durations))
            remaining_epochs = args.epochs - epoch
            eta_sec = avg_epoch * remaining_epochs
            steps_per_sec = steps / epoch_sec if epoch_sec > 0 else float('nan')
            finish_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_sec)
            lr = None
            try:
                lr = optimizer.param_groups[0].get('lr', None)
            except Exception:
                pass
            postfix = {'loss': f'{avg_loss:.6f}', 'epoch': _fmt_hms(epoch_sec), 'avg': _fmt_hms(avg_epoch), 'ETA': _fmt_hms(eta_sec), 'finish': finish_time.strftime('%H:%M:%S'), 's/s': f'{steps_per_sec:.2f}'}
            if lr is not None:
                postfix['lr'] = f'{lr:.6g}'
            pbar.set_postfix(postfix)
        total_train_time = time.time() - global_t0
        print('-' * 80)
        print(f'Training done in {_fmt_hms(total_train_time)}. Now evaluating...')
        embed, y_true = evaluation_network(model, valid_loader, args.pooling, args.device)
        train_score, valid_score, test_score = ogbg_evaluation(embed, y_true, split_idx, evaluator, task_type, args.num_tasks)
        per_round_train.append(float(train_score))
        per_round_valid.append(float(valid_score))
        per_round_test.append(float(test_score))
        final_results.append(float(test_score))
        print(f'[Round {ep_num + 1}/{args.num_exp}] Train={float(train_score):.5f}  Valid={float(valid_score):.5f}  Test={float(test_score):.5f}')
    print('\n' + '=' * 80)
    print('Per-round scores:')
    for i, (tr, va, te) in enumerate(zip(per_round_train, per_round_valid, per_round_test), 1):
        print(f'  Round {i}: Train={tr:.5f}  Valid={va:.5f}  Test={te:.5f}')
    print('-' * 80)
    mean_final = np.mean(per_round_test)
    std_final = np.std(per_round_test)
    print(f'All test scores: {per_round_test}')
    print(f'final result: {mean_final:.5f}±{std_final:.5f}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_exp', type=int, default=5)
    parser.add_argument('--root', type=str, default='./dataset')
    parser.add_argument('--config_dir', type=str, default='./config')
    parser.add_argument('--dataset', type=str, default='ogbg-molbbbp')
    args = parser.parse_args()
    config = load_config(os.path.join(args.config_dir, f'{args.dataset}.yaml'))
    for key, value in config.items():
        setattr(args, key, value)
    args.device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if args.schedule_opt == 'none':
        args.use_scheduler = False
    else:
        args.use_scheduler = True
    print(args)
    main(args)
