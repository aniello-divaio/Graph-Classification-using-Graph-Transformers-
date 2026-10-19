"""
build_dataset_v2.py
Converts bacteria metabolic pathway CSV files into a PyTorch Geometric dataset.
This version uses REAL reaction identifiers (r0, r1, r2...) as node features,
similar to how atom types (C, H, N, O) are used in molecular graphs.

Usage:
    python build_dataset_v2.py \
        --data_dir ~/thesis/data/raw/data/Individuals \
        --labels ~/thesis/data/labels/bacteria_labels.csv \
        --output ~/thesis/data/processed/bacteria_dataset_v2.pt \
        --vocab_output ~/thesis/data/processed/reaction_vocab.json
"""

import os
import argparse
import json
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict


def load_labels(labels_file):
    """Load organism -> class mapping from bacteria_labels.csv"""
    df = pd.read_csv(labels_file)
    labels = {}
    for _, row in df.iterrows():
        labels[str(row['organism'])] = (int(row['ClassIdx']), str(row['Class']))
    print(f"Loaded {len(labels)} organism labels")
    return labels


def extract_base_reaction_id(node_name):
    """
    Extract the base reaction ID, stripping the '_r' (reverse) suffix.
    e.g. 'r340_r' -> 'r340', 'r12' -> 'r12'
    This way the forward and reverse versions of a reaction share
    the same chemical identity, similar to how an atom type is the
    same regardless of its position in a molecule.
    """
    if node_name.endswith('_r'):
        return node_name[:-2]
    return node_name


def build_reaction_vocabulary(data_dir, organism_folders):
    """
    First pass: scan all CSV files to build a vocabulary of all unique
    reaction IDs across the entire dataset. This is analogous to building
    a vocabulary of atom types (C, H, N, O...) before encoding molecules.
    """
    print("Building reaction vocabulary (first pass over all files)...", flush=True)
    all_reactions = set()

    for i, organism in enumerate(organism_folders):
        if i % 1000 == 0:
            print(f"  Scanning {i+1}/{len(organism_folders)}...", flush=True)

        csv_file = os.path.join(data_dir, organism, f"{organism}_R_adj.csv")
        if not os.path.exists(csv_file):
            continue

        try:
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

            for node in list(sources) + list(destinations):
                base_id = extract_base_reaction_id(node)
                all_reactions.add(base_id)
        except Exception:
            continue

    vocab = sorted(all_reactions)
    # Reserve index 0 for unknown/padding, real reactions start at 1
    reaction_to_idx = {r: i + 1 for i, r in enumerate(vocab)}
    print(f"Vocabulary built: {len(reaction_to_idx)} unique reaction types found", flush=True)
    return reaction_to_idx


def load_graph_from_csv(csv_file, reaction_to_idx):
    """
    Load a graph from a CSV edge list file.
    Node features are now the REACTION TYPE (from vocabulary),
    not a placeholder. This is the equivalent of atom type in molecules.
    """
    try:
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

        # Build node features using the reaction vocabulary
        # Each node gets the integer ID of its base reaction type
        node_features = []
        for node in all_nodes:
            base_id = extract_base_reaction_id(node)
            feat = reaction_to_idx.get(base_id, 0)  # 0 = unknown
            node_features.append(feat)

        num_nodes = len(all_nodes)
        x = torch.tensor(node_features, dtype=torch.long).unsqueeze(1)  # [num_nodes, 1]

        return num_nodes, edge_index, x

    except Exception:
        return None, None, None


def build_dataset(data_dir, labels_file, output_file, vocab_output):
    labels = load_labels(labels_file)

    organism_folders = [f for f in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, f))]
    organism_folders.sort()
    print(f"Found {len(organism_folders)} organism folders")

    # First pass: build vocabulary
    reaction_to_idx = build_reaction_vocabulary(data_dir, organism_folders)

    # Save vocabulary for later reference
    with open(vocab_output, 'w') as f:
        json.dump(reaction_to_idx, f)
    print(f"Vocabulary saved to {vocab_output}", flush=True)

    # Second pass: build the actual dataset using the vocabulary
    print("\nBuilding dataset (second pass)...", flush=True)
    dataset = []
    skipped = 0
    not_found = 0
    class_counts = defaultdict(int)

    for i, organism in enumerate(organism_folders):
        if i % 500 == 0:
            print(f"Processing {i+1}/{len(organism_folders)}...", flush=True)

        if organism not in labels:
            not_found += 1
            continue

        class_idx, class_name = labels[organism]

        csv_file = os.path.join(data_dir, organism, f"{organism}_R_adj.csv")
        if not os.path.exists(csv_file):
            skipped += 1
            continue

        num_nodes, edge_index, x = load_graph_from_csv(csv_file, reaction_to_idx)

        if num_nodes is None or num_nodes == 0:
            skipped += 1
            continue

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

    print(f"\nDataset built successfully!", flush=True)
    print(f"  Total graphs: {len(dataset)}", flush=True)
    print(f"  Skipped (no CSV): {skipped}", flush=True)
    print(f"  No label found: {not_found}", flush=True)
    print(f"  Reaction vocabulary size: {len(reaction_to_idx)}", flush=True)
    print(f"\nClass distribution:", flush=True)
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count} graphs", flush=True)

    torch.save(dataset, output_file)
    print(f"\nDataset saved to: {output_file}", flush=True)

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Build bacteria dataset with real reaction node features')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--output', type=str,
                        default='~/thesis/data/processed/bacteria_dataset_v2.pt')
    parser.add_argument('--vocab_output', type=str,
                        default='~/thesis/data/processed/reaction_vocab.json')
    args = parser.parse_args()

    args.output = os.path.expanduser(args.output)
    args.vocab_output = os.path.expanduser(args.vocab_output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    build_dataset(args.data_dir, args.labels, args.output, args.vocab_output)


if __name__ == '__main__':
    main()
