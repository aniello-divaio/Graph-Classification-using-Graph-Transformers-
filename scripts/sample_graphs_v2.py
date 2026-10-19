import argparse
import os

import numpy as np
import torch
from torch_geometric.data import Data


def sample_graph(data, max_nodes=250):
    n = data.num_nodes
    if n <= max_nodes:
        return data
    edge_index = data.edge_index
    degree = torch.bincount(edge_index[0], minlength=n) + \
             torch.bincount(edge_index[1], minlength=n)
    keep = torch.sort(torch.topk(degree, max_nodes).indices).values
    mapping = torch.full((n,), -1, dtype=torch.long)
    mapping[keep] = torch.arange(max_nodes)
    src, dst = mapping[edge_index[0]], mapping[edge_index[1]]
    mask = (src >= 0) & (dst >= 0)
    new_data = Data(
        x=data.x[keep].clone(),          # features preserved
        edge_index=torch.stack([src[mask], dst[mask]]),
        y=data.y,
        num_nodes=max_nodes,
    )
    for attr in ('species', 'class_name'):
        if hasattr(data, attr):
            setattr(new_data, attr, getattr(data, attr))
    return new_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max_nodes', type=int, default=250)
    args = p.parse_args()

    dataset = torch.load(os.path.expanduser(args.dataset), weights_only=False)
    print(f"Loaded {len(dataset)} graphs", flush=True)
    sampled, n_nodes, n_edges = [], [], []
    for i, g in enumerate(dataset):
        s = sample_graph(g, args.max_nodes)
        sampled.append(s)
        n_nodes.append(s.num_nodes)
        n_edges.append(s.edge_index.shape[1])
        if i % 500 == 0:
            print(f"  {i+1}/{len(dataset)}", flush=True)

    torch.save(sampled, os.path.expanduser(args.output))
    const = sum(int(g.x.unique().numel() == 1) for g in sampled)
    print(f"\nSaved to {args.output}")
    print(f"Nodes after sampling: mean={np.mean(n_nodes):.1f} min={min(n_nodes)} max={max(n_nodes)}")
    print(f"Edges after sampling: mean={np.mean(n_edges):.1f} min={min(n_edges)} max={max(n_edges)}")
    print(f"x dtype: {sampled[0].x.dtype} | graphs with constant features: {const}/{len(sampled)} (should be ~0)")


if __name__ == '__main__':
    main()
