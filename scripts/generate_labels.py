"""
generate_labels.py
Builds the class labels of the bacterial organisms from the MetaDAG results
table (Results.csv).

The taxonomic group of each organism is the second field of the KEGG
`Categories` string (e.g. "Bacteria|Actinomycetota|..." -> "Actinomycetota").
The 10 most represented groups are kept as classes and all the others are
merged into "Other" (11 classes in total). Class indices follow alphabetical
order of the class names.

Usage:
    python scripts/generate_labels.py \
        --results Results.csv \
        --output data/labels/bacteria_labels.csv
"""

import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='MetaDAG Results.csv')
    p.add_argument('--output', default='data/labels/bacteria_labels.csv')
    p.add_argument('--top', type=int, default=10,
                   help='number of groups kept as separate classes')
    args = p.parse_args()

    df = pd.read_csv(args.results, usecols=['organism', 'Categories'],
                     low_memory=False)
    df = df[df['Categories'].notna()]
    df['Phylum'] = df['Categories'].apply(
        lambda x: x.split('|')[1] if len(x.split('|')) > 1 else x)

    top_groups = df['Phylum'].value_counts().head(args.top).index.tolist()
    df['Class'] = df['Phylum'].apply(lambda x: x if x in top_groups else 'Other')

    class_names = sorted(df['Class'].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    df['ClassIdx'] = df['Class'].map(class_to_idx)

    labels = df[['organism', 'ClassIdx', 'Class']].copy()
    labels.to_csv(args.output, index=False)
    print(f"Done! {len(labels)} organisms saved to {args.output}")
    print(f"{df['Phylum'].nunique()} groups in total, "
          f"{args.top} kept + 'Other'")
    print(labels['Class'].value_counts())


if __name__ == '__main__':
    main()
