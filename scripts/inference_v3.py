"""
inference_v3.py
Runs Graphormer on the bacteria dataset using REAL reaction node features
(instead of placeholder/degree-based features).

Usage:
    python inference_v3.py \
        --dataset ~/thesis/data/processed/bacteria_dataset_v2_250.pt \
        --output ~/thesis/embeddings/bacteria_embeddings_v2.pt
"""

import torch
import numpy as np
import argparse
import sys
import os

sys.path.insert(0, '/home/aniello/thesis/models/graphormer')
sys.path.insert(0, '/home/aniello/thesis/models/graphormer/graphormer_src')

from graphormer.data.wrapper import preprocess_item
from graphormer.data.collator import collator
from torch_geometric.data import Data

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Manually register Graphormer tasks and models
import graphormer.tasks.graph_prediction
import graphormer.models.graphormer


def convert_to_graphormer_format(data, max_reaction_id=4077):
    """
    Convert a PyG Data object to Graphormer-compatible format.
    This version uses the REAL reaction type ID already stored in data.x,
    instead of computing node degree. This is the equivalent of using
    atom types (C, H, N, O...) in molecular graphs.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # data.x already contains the reaction vocabulary index for each node
    # (built in build_dataset_v2.py). We just need to clamp it to a safe
    # range for the embedding table inside Graphormer.
    reaction_ids = data.x.squeeze(1).clone()
    reaction_ids = reaction_ids.clamp(0, max_reaction_id)

    return Data(
        x=reaction_ids.unsqueeze(1),  # [num_nodes, 1] real reaction type IDs
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
            if i % 200 == 0:
                print(f">>> Graph {i+1}/{len(dataset)} ({data.species})", flush=True)
            try:
                converted = convert_to_graphormer_format(data)
                converted.idx = i
                processed = preprocess_item(converted)
                batched = collator([processed])
                batched_data = {}
                for k, v in batched.items():
                    if isinstance(v, torch.Tensor):
                        if k == 'x':
                            v = v.long()
                        batched_data[k] = v.to(device)
                    else:
                        batched_data[k] = v

                # Hook to extract 768-dim embedding before final projection
                embedding_768 = {}
                def hook_fn(module, input, output):
                    embedding_768['value'] = output
                hook = model.encoder.layer_norm.register_forward_hook(hook_fn)
                output = model(batched_data)
                hook.remove()

                graph_embedding = embedding_768['value'][0, :].cpu()  # [251, 768]
                graph_embedding = graph_embedding[0, :]  # VNode -> [768]

                all_embeddings.append(graph_embedding)
                all_labels.append(data.y.item())
                all_species.append(data.species)
            except Exception as e:
                logger.warning(f"  ERROR on {data.species}: {e}")
                continue
            if i % 50 == 0:
                torch.cuda.empty_cache()

    embeddings = torch.stack(all_embeddings)
    labels = torch.tensor(all_labels)
    logger.info(f"Done! Embeddings shape: {embeddings.shape}")
    return embeddings, labels, all_species


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='/home/aniello/thesis/data/processed/bacteria_dataset_v2_250.pt')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/aniello/.cache/torch/hub/checkpoints/checkpoint_best_pcqm4mv1.pt')
    parser.add_argument('--output', type=str,
                        default='/home/aniello/thesis/embeddings/bacteria_embeddings_v2.pt')
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
    for c in sorted(set(labels.tolist())):
        count = (labels == c).sum().item()
        print(f"  Class {c}: {count} graphs", flush=True)


if __name__ == '__main__':
    main()
