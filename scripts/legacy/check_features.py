import argparse
import os
from collections import Counter

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full', default='~/thesis/data/processed/bacteria_dataset_v2.pt')
    p.add_argument('--sampled', default='~/thesis/data/processed/bacteria_dataset_v2_250.pt')
    args = p.parse_args()

    full = torch.load(os.path.expanduser(args.full), weights_only=False)
    sampled = torch.load(os.path.expanduser(args.sampled), weights_only=False)
    print(f"Graphs: full={len(full)}  sampled={len(sampled)}")

    sizes = [g.num_nodes for g in full]
    small = [g for g in full if g.num_nodes <= 250]
    print(f"\nNodes before sampling: mean={sum(sizes)/len(sizes):.1f} "
          f"min={min(sizes)} max={max(sizes)}")
    print(f"Graphs with <= 250 nodes (left untouched by sampling): {len(small)}")
    cls = Counter(getattr(g, 'class_name', int(g.y)) for g in small)
    for k, v in cls.most_common():
        print(f"   {k}: {v}")

    dtypes = Counter(str(g.x.dtype) for g in sampled)
    constant = sum(int(g.x.unique().numel() == 1) for g in sampled)
    print(f"\nSampled dataset: x dtypes = {dict(dtypes)}")
    print(f"Graphs where ALL nodes have the same feature: {constant} / {len(sampled)}")
    print(f"First graph, first 10 features: {sampled[0].x[:10].flatten().tolist()}")

    floats = sum(int(g.x.is_floating_point()) for g in sampled)
    if constant > len(sampled) * 0.05 or floats > 0:
        print(f"\n>>> PROBLEM CONFIRMED: {constant} graphs "
              f"({100*constant/len(sampled):.1f}%) have constant features and "
              f"{floats} have float features (= torch.ones from sample_graphs.py). "
              "The reaction IDs were lost: the v2 result must be recomputed.")
    else:
        print("\n>>> Reaction IDs look preserved. The v2 result is valid.")


if __name__ == '__main__':
    main()
