"""
build_dataset_bacteria.py
Converts bacteria metabolic pathway CSV files into a PyTorch Geometric dataset.
Handles the new MetaDAG folder structure: Individuals/organism/organism_R_adj.csv

Usage:
    python build_dataset_bacteria.py \
        --data_dir ~/thesis/data/raw/data/Individuals \
        --labels ~/thesis/data/labels/bacteria_labels.csv \
        --output ~/thesis/data/processed/bacteria_dataset.pt
"""

import os
import argparse
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict


def load_labels(labels_file):
    """
    Load organism -> class mapping from bacteria_labels.csv
    Returns a dict: {organism_code: (class_idx, class_name)}
    """
    df = pd.read_csv(labels_file)
    labels = {}
    for _, row in df.iterrows():
        labels[str(row['organism'])] = (int(row['ClassIdx']), str(row['Class']))
    print(f"Loaded {len(labels)} organism labels")
    return labels


def load_graph_from_csv(csv_file):
    """
    Load a graph from a CSV edge list file.
    Handles both comma and semicolon separators.
    Returns (num_nodes, edge_index) or (None, None) on error.
    """
    try:
        # Try semicolon first, then comma
        for sep in [';', ',']:
            try:
                df = pd.read_csv(csv_file, sep=sep, quotechar='"')
                if 'source' in df.columns and 'destination' in df.columns:
                    break
            except:
                continue

        df.columns = df.columns.str.strip()
        sources = df['source'].astype(str).str.strip().str.replace('"', '')
        destinations = df['destination'].astype(str).str.strip().str.replace('"', '')

        all_nodes = sorted(set(sources.tolist() + destinations.tolist()))
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

        src_indices = [node_to_idx[s] for s in sources]
        dst_indices = [node_to_idx[d] for d in destinations]

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        num_nodes = len(all_nodes)

        return num_nodes, edge_index

    except Exception as e:
        return None, None


def build_dataset(data_dir, labels_file, output_file):
    """
    Build the full dataset from all organism folders.
    """
    labels = load_labels(labels_file)

    # Find all organism folders
    organism_folders = [f for f in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, f))]
    organism_folders.sort()
    print(f"Found {len(organism_folders)} organism folders")

    dataset = []
    skipped = 0
    not_found = 0
    class_counts = defaultdict(int)

    for i, organism in enumerate(organism_folders):
        if i % 500 == 0:
            print(f"Processing {i+1}/{len(organism_folders)}...", flush=True)

        # Check if we have a label for this organism
        if organism not in labels:
            not_found += 1
            continue

        class_idx, class_name = labels[organism]

        # Find the R_adj.csv file inside the organism folder
        organism_dir = os.path.join(data_dir, organism)
        csv_file = os.path.join(organism_dir, f"{organism}_R_adj.csv")

        if not os.path.exists(csv_file):
            skipped += 1
            continue

        # Load graph
        num_nodes, edge_index = load_graph_from_csv(csv_file)

        if num_nodes is None or num_nodes == 0:
            skipped += 1
            continue

        # Create node features (all ones placeholder)
        x = torch.ones((num_nodes, 1), dtype=torch.float)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([class_idx], dtype=torch.long),
            num_nodes=num_nodes,
            species=organism,
            class_name=class_name
        )

        dataset.append(data)
        class_counts[class_name] += 1

    print(f"\nDataset built successfully!")
    print(f"  Total graphs: {len(dataset)}")
    print(f"  Skipped (no CSV): {skipped}")
    print(f"  No label found: {not_found}")
    print(f"\nClass distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count} graphs")

    # Print graph statistics
    node_counts = [d.num_nodes for d in dataset]
    edge_counts = [d.edge_index.shape[1] for d in dataset]
    print(f"\nGraph statistics:")
    print(f"  Avg nodes per graph: {sum(node_counts)/len(node_counts):.1f}")
    print(f"  Min nodes: {min(node_counts)}, Max nodes: {max(node_counts)}")
    print(f"  Avg edges per graph: {sum(edge_counts)/len(edge_counts):.1f}")
    print(f"  Min edges: {min(edge_counts)}, Max edges: {max(edge_counts)}")

    torch.save(dataset, output_file)
    print(f"\nDataset saved to: {output_file}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Build bacteria metabolic pathway dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing organism subfolders (Individuals/)')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to bacteria_labels.csv')
    parser.add_argument('--output', type=str,
                        default='~/thesis/data/processed/bacteria_dataset.pt',
                        help='Output dataset file')
    args = parser.parse_args()

    args.output = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    build_dataset(args.data_dir, args.labels, args.output)


if __name__ == '__main__':
    main()
