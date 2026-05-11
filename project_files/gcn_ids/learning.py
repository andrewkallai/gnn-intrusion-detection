import argparse
import os
import os.path as osp
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader as PyGNeighborLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    import cupy
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    from rmm.allocators.torch import rmm_torch_allocator
    rmm.reinitialize(devices=[0], pool_allocator=True, managed_memory=True)
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    import cudf  # noqa
    import cugraph_pyg  # noqa
    cudf.set_option("spill", True)

import torch.nn.functional as F  # noqa
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from ogb.nodeproppred import PygNodePropPredDataset  # noqa


import torch_geometric  # noqa


class IDSWindowedDataset:
    """Dataset class for loading windowed graph data from data_graph.py pipeline.
    
    Loads .npz files containing:
    - node_features: Node feature matrix
    - node_labels: Node labels (0=benign, 1=malicious)
    - edge_index: Edge index in COO format (2 x num_edges)
    - edge_features: Edge feature matrix
    """
    
    def __init__(self, root_dir: str, split: str = "train"):
        """
        Args:
            root_dir: Root directory containing graphs/{split}/ subdirectories
            split: Split name ('train', 'val', or 'test')
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.graphs_dir = self.root_dir / "graphs" / split
        
        if not self.graphs_dir.exists():
            raise ValueError(f"Graph directory not found: {self.graphs_dir}")
        
        # Load all .npz files in the split directory
        self.graph_files = sorted(self.graphs_dir.glob("*.npz"))
        if not self.graph_files:
            raise ValueError(f"No .npz files found in {self.graphs_dir}")
        
        # Load the first graph as the main data object
        # For windowed data, we typically train on one window at a time
        # or concatenate all windows into a single graph
        self._load_graphs()
    
    def _load_graphs(self):
        """Load all graphs from .npz files and combine into a single Data object."""
        all_node_features = []
        all_node_labels = []
        all_edge_indices = []
        all_edge_features = []
        node_offset = 0
        
        for graph_file in self.graph_files:
            data = np.load(graph_file)
            
            num_nodes = data['node_features'].shape[0]
            
            all_node_features.append(data['node_features'])
            all_node_labels.append(data['node_labels'])
            
            # Adjust edge indices for node offset
            edge_index = data['edge_index'].astype(np.int64).copy()
            edge_index[:] += node_offset
            all_edge_indices.append(edge_index)
            
            all_edge_features.append(data['edge_features'])
            node_offset += num_nodes
        
        # Concatenate all arrays
        self.node_features = np.vstack(all_node_features)
        self.node_labels = np.concatenate(all_node_labels)
        
        # IMPORTANT: Fixed edge concatenation
        # edge_index is (2, num_edges). We need to stack along the second axis.
        self.edge_index = np.hstack(all_edge_indices)
        
        self.edge_features = np.vstack(all_edge_features) if all_edge_features[0].shape[0] > 0 else np.zeros((0, 10))
        
        # Convert to torch tensors
        self.x = torch.tensor(self.node_features, dtype=torch.float32)
        self.y = torch.tensor(self.node_labels, dtype=torch.long)
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        self.edge_attr = torch.tensor(self.edge_features, dtype=torch.float32) if self.edge_features.shape[0] > 0 else torch.zeros((0, 10), dtype=torch.float32)
        
        self.num_nodes = self.x.shape[0]
        self.num_features = self.x.shape[1]
        self.num_classes = 2  # Binary classification: benign vs malicious
        
        # Create split indices (all nodes in this split are used)
        self.split_idx = {
            'train': torch.arange(self.num_nodes) if self.split == 'train' else torch.tensor([], dtype=torch.long),
            'valid': torch.arange(self.num_nodes) if self.split == 'val' else torch.tensor([], dtype=torch.long),
            'test': torch.arange(self.num_nodes) if self.split == 'test' else torch.tensor([], dtype=torch.long),
        }
    
    def __getitem__(self, idx):
        """Return a PyG Data object."""
        data = Data()
        data.x = self.x
        data.y = self.y
        data.edge_index = self.edge_index
        data.edge_attr = self.edge_attr
        data.num_nodes = self.num_nodes
        return data
    
    def __len__(self):
        return 1
    
    def get_idx_split(self):
        """Return split indices."""
        return self.split_idx
    
    @property
    def num_graphs(self):
        return len(self.graph_files)


# ---------------- Distributed helpers ----------------
def safe_get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def safe_get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def init_distributed():
    """Initialize distributed training if environment variables are set.
    Fallback to single-process mode otherwise.

    Returns the device_id to use for barrier() calls.
    """
    if dist.is_available() and dist.is_initialized():
        return 0

    default_env = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500"
    }

    for k, v in default_env.items():
        os.environ.setdefault(k, v)

    device_id = 0
    if torch.cuda.is_available():
        device_id = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(device_id)

    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", device_id=device_id)
        rank = os.environ['RANK']
        print(f"Initialized distributed: rank {rank}, world_size {world_size}, backend={backend}")
    else:
        print("Running in single-process mode (CPU)" if not torch.cuda.is_available() else "Running in single-GPU / single-process mode")

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=0,
                                world_size=1, device_id=device_id)

    return device_id


# ------------------------------------------------------


def arg_parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ids-custom',
        choices=['ogbn-papers100M', 'ogbn-products', 'ogbn-arxiv', 'ids-custom', 'ids-unsw-full'],
        help='Dataset name. Use "ids-custom" or "ids-unsw-full" for custom IDS data.',
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/workspace/data',
        help='Root directory of dataset.',
    )
    parser.add_argument(
        "--dataset_subdir",
        type=str,
        default="",
        help="Subdirectory of dataset (for OGB datasets). Ignored for ids-custom.",
    )
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('--fan_out', type=int, default=10)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument(
        '--use_directed_graph',
        action='store_true',
        help='Whether or not to use directed graph',
    )
    parser.add_argument(
        '--add_self_loop',
        action='store_true',
        help='Whether or not to add self loop',
    )
    parser.add_argument(
        "--model",
        type=str,
        default='GCN',
        choices=[
            'SAGE',
            'GAT',
            'GCN',
            # TODO: Uncomment when we add support for disjoint sampling
            # 'SGFormer',
        ],
        help="Model used for training, default SAGE",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="If using GATConv or GT, number of attention heads to use",
    )
    parser.add_argument('--tempdir_root', type=str, default=None)
    args = parser.parse_args()
    return args


def create_loader(
    input_nodes,
    stage_name,
    data,
    num_neighbors,
    replace,
    batch_size,
    shuffle=False,
):
    if safe_get_rank() == 0:
        print(f'Creating {stage_name} loader...')

    return PyGNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        replace=replace,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train(model, train_loader, optimizer, is_custom_dataset=False):
    """Train for one epoch.

    Returns the average loss and accuracy for the epoch. The implementation
    normalises by the actual number of samples processed and converts all
    tensors to Python scalars to avoid accidental tensor‑scalar mixing.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in train_loader:
        if is_custom_dataset:
            x = batch.x.to(DEVICE)
            y = batch.y.to(DEVICE)
            edge_index = batch.edge_index.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.cross_entropy(out, y.view(-1))
            loss.backward()
            optimizer.step()

            batch_sz = y.size(0)
            total_loss += loss.item() * batch_sz
            total_correct += out.argmax(dim=-1).eq(y.view(-1)).sum().item()
            total_examples += batch_sz
        else:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[: batch.batch_size]
            y = batch.y[: batch.batch_size].view(-1).to(torch.long)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            batch_sz = y.size(0)
            total_loss += loss.item() * batch_sz
            total_correct += out.argmax(dim=-1).eq(y).sum().item()
            total_examples += batch_sz

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test(model, loader, is_custom_dataset=False):
    model.eval()

    total_correct = total_examples = 0
    for batch in loader:
        if is_custom_dataset:
            batch_x = batch.x.to(DEVICE)
            batch_y = batch.y.to(DEVICE)
            batch_edge_index = batch.edge_index.to(DEVICE)
            
            out = model(batch_x, batch_edge_index)
            total_correct += out.argmax(dim=-1).eq(batch_y.view(-1)).sum().item()
            total_examples += batch_y.size(0)
        else:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].view(-1).to(torch.long)

            total_correct += out.argmax(dim=-1).eq(y).sum()
            total_examples += y.size(0)

    return total_correct / total_examples

# ---------------------------------------------------------------------
# Visualization helpers (confusion matrix)
# ---------------------------------------------------------------------
def _gather_predictions(model, loader, is_custom_dataset=False):
    """Run ``model`` on ``loader`` and collect predictions and true labels.

    Returns two NumPy arrays: ``preds`` and ``labels``. Used after training to
    compute a confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        if is_custom_dataset:
            x = batch.x.to(DEVICE)
            y = batch.y.to(DEVICE)
            edge_index = batch.edge_index.to(DEVICE)
            out = model(x, edge_index)
            preds = out.argmax(dim=-1).cpu().numpy()
            labels = y.view(-1).cpu().numpy()
        else:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index)[: batch.batch_size]
            y = batch.y[: batch.batch_size].view(-1).to(torch.long)
            preds = out.argmax(dim=-1).cpu().numpy()
            labels = y.cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)
    return np.concatenate(all_preds), np.concatenate(all_labels)

def _plot_confusion(cm, class_names, save_path="confusion_matrix.png"):
    """Plot ``cm`` (confusion matrix) and save to ``save_path``.

    The matrix is visualised with a blue colormap and cell counts are
    annotated.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xlim=[-0.5, len(class_names) - 0.5],
        ylim=[-0.5, len(class_names) - 0.5],
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    # init DDP if needed
    device_id = init_distributed()

    args = arg_parse()
    torch_geometric.seed_everything(123)

    if "papers" in str(args.dataset) and (psutil.virtual_memory().total /
                                          (1024**3)) < 390:
        if safe_get_rank() == 0:
            print("Warning: may not have enough RAM to use this many GPUs.")
            print("Consider upgrading RAM if an error occurs.")
            print("Estimated RAM Needed: ~390GB.")

    wall_clock_start = time.perf_counter()

    if args.dataset in ['ids-custom', 'ids-unsw-full']:
        # Load custom IDS dataset from data_graph.py pipeline
        dataset_root = Path(args.dataset_dir)
        
        # If using unsw-full, we definitely want the subdirectory
        if args.dataset == 'ids-unsw-full':
            dataset_root = dataset_root / 'graph_unsw_full_10min'
        elif args.dataset == 'ids-custom':
            # Default path for ids-custom as requested
            custom_path = dataset_root / 'graph_10min_moduleA_stratified'
            if custom_path.exists():
                dataset_root = custom_path
            else:
                # Fallback: check if we are at the parent level or already in the folder
                if not (dataset_root / "graphs").exists():
                    unsw_path = dataset_root / 'graph_unsw_full_10min'
                    if (unsw_path / "graphs").exists():
                        dataset_root = unsw_path

        if safe_get_rank() == 0:
            print(f'Loading custom IDS dataset from: {dataset_root}')
        
        # Load train, val, test splits
        train_dataset = IDSWindowedDataset(str(dataset_root), split='train')
        val_dataset = IDSWindowedDataset(str(dataset_root), split='val')
        test_dataset = IDSWindowedDataset(str(dataset_root), split='test')
        
        # Get data objects
        train_data = train_dataset[0]
        val_data = val_dataset[0]
        test_data = test_dataset[0]
        
        # Use train data as the main data object
        data = train_data
        
        if safe_get_rank() == 0:
            print(f'Loaded {args.dataset} dataset:')
            print(f'  Train: {train_data.num_nodes} nodes, {train_data.edge_index.shape[1]} edges')
            print(f'  Val: {val_data.num_nodes} nodes, {val_data.edge_index.shape[1]} edges')
            print(f'  Test: {test_data.num_nodes} nodes, {test_data.edge_index.shape[1]} edges')
            print(f'  Node features: {train_data.x.shape[1]}, Classes: {train_dataset.num_classes}')
        
        # Create split indices for training
        split_idx = {
            'train': torch.arange(train_data.num_nodes),
            'valid': torch.arange(val_data.num_nodes),
            'test': torch.arange(test_data.num_nodes),
        }
        
        # Store datasets for later use
        datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        
    else:
        root = osp.join(args.dataset_dir, args.dataset_subdir)

        if safe_get_rank() == 0:
            print('The root is: ', root)

        torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
        torch.serialization.add_safe_globals([torch_geometric.data.data.DataTensorAttr])
        torch.serialization.add_safe_globals([torch_geometric.data.storage.GlobalStorage])

        dataset = PygNodePropPredDataset(name=args.dataset, root=root)
        split_idx = dataset.get_idx_split()

        data = dataset[0]
        if not args.use_directed_graph:
            data.edge_index = torch_geometric.utils.to_undirected(
                data.edge_index, reduce="mean")
        if args.add_self_loop:
            data.edge_index, _ = torch_geometric.utils.remove_self_loops(
                data.edge_index)
            data.edge_index, _ = torch_geometric.utils.add_self_loops(
                data.edge_index, num_nodes=data.num_nodes)
        
        if torch.cuda.is_available():
            graph_store = cugraph_pyg.data.GraphStore()
            graph_store[dict(
                edge_type=('node', 'rel', 'node'),
                layout='coo',
                is_sorted=False,
                size=(data.num_nodes, data.num_nodes),
            )] = data.edge_index

            feature_store = cugraph_pyg.data.FeatureStore()
            feature_store['node', 'x', None] = data.x
            feature_store['node', 'y', None] = data.y

            data = (feature_store, graph_store)
        datasets = None

    if safe_get_rank() == 0:
        print(f"Training {args.dataset} with {args.model} model.")

    # Get dataset info based on dataset type
    if args.dataset in ['ids-custom', 'ids-unsw-full']:
        num_features = data.x.shape[1]
        num_classes = 2  # Binary classification
    else:
        num_features = dataset.num_features
        num_classes = dataset.num_classes

    if args.model == "GAT":
        model = torch_geometric.nn.models.GAT(num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              num_classes,
                                              heads=args.num_heads).to(DEVICE)
    elif args.model == "GCN":
        model = torch_geometric.nn.models.GCN(num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              num_classes).to(DEVICE)
    elif args.model == "SAGE":
        model = torch_geometric.nn.models.GraphSAGE(
            num_features, args.hidden_channels, args.num_layers,
            num_classes).to(DEVICE)
    elif args.model == 'SGFormer':
        model = torch_geometric.nn.models.SGFormer(
            in_channels=num_features,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            trans_num_heads=args.num_heads,
            trans_dropout=args.dropout,
            gnn_num_layers=args.num_layers,
            gnn_dropout=args.dropout,
        ).to(DEVICE)
    else:
        raise ValueError(f'Unsupported model type: {args.model}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.wd)

    if args.dataset in ['ids-custom', 'ids-unsw-full']:
        # For custom IDS dataset, use PyG's NeighborLoader for neighbor sampling
        # This properly handles sampling neighbors for each node in the batch
        
        num_neighbors = [args.fan_out] * args.num_layers
        
        # Create NeighborLoaders for each split
        # The loader will sample neighbors for nodes in each batch
        train_loader = PyGNeighborLoader(
            data=data,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = PyGNeighborLoader(
            data=val_data,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        test_loader = PyGNeighborLoader(
            data=test_data,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
    else:
        if torch.cuda.is_available():
            graph_store = cugraph_pyg.data.GraphStore()
            graph_store[dict(
                edge_type=('node', 'rel', 'node'),
                layout='coo',
                is_sorted=False,
                size=(data.num_nodes, data.num_nodes),
            )] = data.edge_index

            feature_store = cugraph_pyg.data.FeatureStore()
            feature_store['node', 'x', None] = data.x
            feature_store['node', 'y', None] = data.y

            data = (feature_store, graph_store)

        loader_kwargs = dict(
            data=data,
            num_neighbors=[args.fan_out] * args.num_layers,
            replace=False,
            batch_size=args.batch_size,
        )

        train_loader = create_loader(split_idx['train'], 'train', **loader_kwargs,
                                     shuffle=True)
        val_loader = create_loader(split_idx['valid'], 'val', **loader_kwargs)
        test_loader = create_loader(split_idx['test'], 'test', **loader_kwargs)

    if dist.is_initialized():
        dist.barrier()  # sync before training

    # Determine if using custom dataset
    is_custom_dataset = (args.dataset in ['ids-custom', 'ids-unsw-full'])

    if safe_get_rank() == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time) =", prep_time,
              "seconds")
        print("Beginning training...")

    val_accs, times, train_times, inference_times = [], [], [], []
    best_val = 0.
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_start = time.perf_counter()
        loss, train_acc = train(model, train_loader, optimizer, is_custom_dataset=is_custom_dataset)
        train_end = time.perf_counter()
        train_times.append(train_end - train_start)
        inference_start = time.perf_counter()
        train_acc = test(model, train_loader, is_custom_dataset=is_custom_dataset)
        val_acc = test(model, val_loader, is_custom_dataset=is_custom_dataset)
        inference_times.append(time.perf_counter() - inference_start)
        val_accs.append(val_acc)

        if safe_get_rank() == 0:
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Time: {train_end - train_start:.4f}s')

        times.append(time.perf_counter() - train_start)
        best_val = max(best_val, val_acc)

    if safe_get_rank() == 0:
        print(f"Total time used: {time.perf_counter()-start:.4f}")
        print("Final Validation: {:.4f} ± {:.4f}".format(
            torch.tensor(val_accs).mean(),
            torch.tensor(val_accs).std()))
        print(f"Best validation accuracy: {best_val:.4f}")
        print("Testing...")
        final_test_acc = test(model, test_loader, is_custom_dataset=is_custom_dataset)
        preds, labels = _gather_predictions(model, test_loader,
                                            is_custom_dataset=is_custom_dataset)
        cm = confusion_matrix(labels, preds)
        precision = precision_score(labels, preds, average='binary')
        recall = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')

        class_names = ['Benign', 'Malicious'] if args.dataset in ['ids-custom', 'ids-unsw-full'] else [str(i) for i in range(cm.shape[0])]
        
        cm_filename = f"confusion_matrix_{args.dataset}_{args.epochs}epochs.png"
        _plot_confusion(cm, class_names, save_path=cm_filename)

        test_acc = (preds == labels).mean()

        metrics_report = (
            f"Dataset: {args.dataset}\n"
            f"Model: {args.model}\n"
            f"Epochs: {args.epochs}\n"
            f"----------------------------\n"
            f"Test Accuracy: {test_acc:.4f}\n"
            f"Test Precision: {precision:.4f}\n"
            f"Test Recall: {recall:.4f}\n"
            f"Test F1 Score: {f1:.4f}\n"
        )

        print(metrics_report)
        
        metrics_filename = f"metrics_{args.dataset}_{args.epochs}epochs.txt"
        with open(metrics_filename, "w") as f:
            f.write(metrics_report)

        preds_dir = Path(f"predictions_{args.dataset}")
        preds_dir.mkdir(exist_ok=True)
        
        test_graphs_dir = test_dataset.graphs_dir
        test_graph_files = sorted(test_graphs_dir.glob("*.npz"))
        node_offset = 0
        for w, graph_file in enumerate(test_graph_files):
            data_win = np.load(graph_file)
            num_nodes_win = data_win['node_features'].shape[0]
            win_preds = preds[node_offset:node_offset + num_nodes_win]
            win_labels = labels[node_offset:node_offset + num_nodes_win]
            np.save(preds_dir / f"window_{w:05d}_preds.npy", win_preds)
            np.save(preds_dir / f"window_{w:05d}_labels.npy", win_labels)
            node_offset += num_nodes_win
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total Program Runtime (total_time) =", total_time, "seconds")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
