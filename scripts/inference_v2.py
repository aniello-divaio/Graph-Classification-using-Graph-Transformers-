"""
inference_v2.py
Runs Graphormer on the archaea metabolic pathway dataset and extracts embeddings.
This version avoids triggering any dataset downloads.

Usage:
    python inference_v2.py \
        --dataset /home/aniello/Graphormer/archaea_dataset.pt \
        --output /home/aniello/Graphormer/archaea_embeddings.pt
"""

import torch
import numpy as np
import argparse
import sys
import os

sys.path.insert(0, '/home/aniello/thesis/models/graphormer')
sys.path.insert(0, '/home/aniello/thesis/models/graphormer/graphormer_src')

from graphormer.data.wrapper import preprocess_item

# Manually register Graphormer tasks and models
import graphormer.tasks.graph_prediction
import graphormer.models.graphormer
from graphormer.data.collator import collator
from torch_geometric.data import Data

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def convert_to_graphormer_format(data):
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    degree = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index.shape[1] > 0:
        for s in edge_index[0]:
            degree[s] += 1
    degree = degree.clamp(0, 512)
    return Data(
        x=degree.unsqueeze(1),
        edge_index=edge_index,
        edge_attr=torch.zeros(edge_index.shape[1], 1, dtype=torch.long),
        y=data.y,
        num_nodes=num_nodes
    )


def load_model(checkpoint_path):
    from fairseq import options, tasks
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf

    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model']

    parser = options.get_training_parser()
    args = options.parse_args_and_arch(
        parser,
        input_args=[
            '--task', 'graph_prediction',
            '--dataset-name', 'pcqm4m',
            '--dataset-source', 'ogb',
            '--arch', 'graphormer_base',
            '--num-classes', '1',
            '--ddp-backend=legacy_ddp',
            '--no-progress-bar',
        ],
        modify_parser=None
    )
    cfg = convert_namespace_to_omegaconf(args)

    # Monkey-patch to prevent dataset loading
    import graphormer.tasks.graph_prediction as gp
    original_init = gp.GraphPredictionTask.__init__
    def patched_init(self, cfg):
        self.cfg = cfg
        self.dm = None
    gp.GraphPredictionTask.__init__ = patched_init

    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # Restore original init
    gp.GraphPredictionTask.__init__ = original_init

    logger.info("Model loaded!")
    return model


def extract_embeddings(model, dataset, device):
    all_embeddings = []
    all_labels = []
    all_species = []

    model = model.to(device)
    logger.info(f"Extracting embeddings for {len(dataset)} graphs...")

    with torch.no_grad():
        for i, data in enumerate(dataset):
            print(f">>> Graph {i+1}/{len(dataset)} ({data.species})", flush=True)
            try:
                converted = convert_to_graphormer_format(data)
                converted.idx = i
                processed = preprocess_item(converted)
                batched = collator([processed])
                # Move to device
                batched_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batched.items()}
                # Hook to extract 768-dim embedding before final projection
                embedding_768 = {}
                def hook_fn(module, input, output):
                    embedding_768['value'] = output
                hook = model.encoder.layer_norm.register_forward_hook(hook_fn)
                output = model(batched_data)
                hook.remove()
                # Get the [VNode] embedding (position 0) with full 768 dims
                graph_embedding = embedding_768['value'][0, :].cpu()  # [251, 768]
                graph_embedding = graph_embedding[0, :]  # take [VNode] at position 0 -> [768]
                all_embeddings.append(graph_embedding)
                all_labels.append(data.y.item())
                all_species.append(data.species)
            except Exception as e:
                logger.warning(f"  ERROR on {data.species}: {e}")
                continue
            if i % 10 == 0:
                torch.cuda.empty_cache()

    embeddings = torch.stack(all_embeddings)
    labels = torch.tensor(all_labels)
    logger.info(f"Done! Embeddings shape: {embeddings.shape}")
    return embeddings, labels, all_species


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='/home/aniello/Graphormer/archaea_dataset.pt')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/aniello/.cache/torch/hub/checkpoints/checkpoint_best_pcqm4mv1.pt')
    parser.add_argument('--output', type=str,
                        default='/home/aniello/Graphormer/archaea_embeddings.pt')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda')
    logger.info(f"Using device: {device}")

    dataset = torch.load(args.dataset)
    logger.info(f"Loaded {len(dataset)} graphs")

    model = load_model(args.checkpoint)
    embeddings, labels, species = extract_embeddings(model, dataset, device)

    torch.save({'embeddings': embeddings, 'labels': labels, 'species': species}, args.output)
    logger.info(f"Saved to: {args.output}")

    print("\n=== SUMMARY ===", flush=True)
    print(f"Graphs processed: {len(species)}", flush=True)
    print(f"Embedding shape:  {embeddings.shape}", flush=True)
    class_names = {0:'6010', 1:'6011', 2:'6013', 3:'6014', 4:'6015', 5:'6016'}
    for c in range(6):
        count = (labels == c).sum().item()
        print(f"  Class {c} (KEGG {class_names[c]}): {count} graphs", flush=True)


if __name__ == '__main__':
    main()
