"""
sample_graphs.py
Reduces each graph in the dataset to a maximum of 250 nodes
by keeping the highest-degree nodes (most connected reactions).

Usage:
    python sample_graphs.py \
        --dataset /home/aniello/Graphormer/archaea_dataset.pt \
        --output /home/aniello/Graphormer/archaea_dataset_250.pt \
        --max_nodes 250
"""

import torch
import argparse
import numpy as np
from torch_geometric.data import Data


def sample_graph(data, max_nodes=250):
    """
    Reduce a graph to max_nodes by keeping the highest-degree nodes.
    Edges are kept only if both endpoints are in the selected nodes.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    if num_nodes <= max_nodes:
        return data

    # Compute degree for each node (in + out)
    degree = torch.zeros(num_nodes, dtype=torch.long)
    for s in edge_index[0]:
        degree[s] += 1
    for d in edge_index[1]:
        degree[d] += 1

    # Select top max_nodes by degree
    top_nodes = torch.topk(degree, max_nodes).indices
    top_nodes_set = set(top_nodes.tolist())

    # Build remapping: old index -> new index
    node_map = {old: new for new, old in enumerate(sorted(top_nodes_set))}

    # Filter edges where both endpoints are in selected nodes
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    new_src, new_dst = [], []
    for s, d in zip(src, dst):
        if s in top_nodes_set and d in top_nodes_set:
            new_src.append(node_map[s])
            new_dst.append(node_map[d])

    new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)

    # New node features (degree of selected nodes)
    new_x = torch.ones((max_nodes, 1), dtype=torch.float)

    return Data(
        x=new_x,
        edge_index=new_edge_index,
        y=data.y,
        num_nodes=max_nodes,
        species=data.species
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='/home/aniello/Graphormer/archaea_dataset.pt')
    parser.add_argument('--output', type=str,
                        default='/home/aniello/Graphormer/archaea_dataset_250.pt')
    parser.add_argument('--max_nodes', type=int, default=250)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...", flush=True)
    dataset = torch.load(args.dataset)
    print(f"Loaded {len(dataset)} graphs", flush=True)

    sampled = []
    edge_counts = []

    for i, data in enumerate(dataset):
        new_data = sample_graph(data, args.max_nodes)
        sampled.append(new_data)
        edge_counts.append(new_data.edge_index.shape[1])
        print(f">>> {i+1}/{len(dataset)} ({data.species}): "
              f"{data.num_nodes} nodes -> {new_data.num_nodes} nodes, "
              f"{new_data.edge_index.shape[1]} edges", flush=True)

    torch.save(sampled, args.output)

    print(f"\nDone! Saved to {args.output}", flush=True)
    print(f"All graphs now have exactly {args.max_nodes} nodes", flush=True)
    print(f"Avg edges after sampling: {np.mean(edge_counts):.1f}", flush=True)
    print(f"Min edges: {min(edge_counts)}, Max edges: {max(edge_counts)}", flush=True)


if __name__ == '__main__':
    main()
