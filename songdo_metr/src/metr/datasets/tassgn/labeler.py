"""
Labeler for Phase 2: Binary clustering to generate pattern labels.

This module implements the hierarchical binary clustering algorithm
that generates cluster labels for future time series patterns.
Labels are used to train the Predictor in Phase 3.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Labeler:
    """Hierarchical binary clustering for traffic pattern labeling.
    
    The Labeler performs Algorithm 1 (Binary Cluster) from the TASSGN paper:
    1. If samples <= threshold, assign single label
    2. Otherwise, split into 2 clusters using MiniBatchKMeans
    3. Recursively apply to each cluster
    
    This creates a hierarchical label structure where similar future
    patterns share the same label.
    
    Args:
        num_nodes: Number of sensor nodes
        thresh: Minimum cluster size threshold (default: 30)
        repeat: Number of clustering repetitions for best result (default: 10)
        
    Example:
        ```python
        labeler = Labeler(num_nodes=170, thresh=30, repeat=10)
        train_labels, val_labels = labeler.generate_labels(
            train_repr, val_repr
        )
        np.save("train_cluster_label.npy", train_labels)
        np.save("val_cluster_label.npy", val_labels)
        ```
    """
    
    def __init__(
        self, 
        num_nodes: int, 
        thresh: int = 30, 
        repeat: int = 10
    ):
        self.num_nodes = num_nodes
        self.thresh = thresh
        self.repeat = repeat
        self.label_counter = 0
    
    def binary_cluster(
        self,
        train_x: np.ndarray,
        train_index: np.ndarray,
        train_labels: np.ndarray,
        val_x: np.ndarray,
        val_index: np.ndarray,
        val_labels: np.ndarray,
    ) -> None:
        """Recursively apply binary clustering (Algorithm 1).
        
        This method modifies train_labels and val_labels in-place.
        
        Args:
            train_x: Training representations, shape (n_train, hidden_dim)
            train_index: Indices into train_labels array
            train_labels: Array to store generated labels (modified in-place)
            val_x: Validation representations, shape (n_val, hidden_dim)
            val_index: Indices into val_labels array
            val_labels: Array to store generated labels (modified in-place)
        """
        # Base case: small cluster
        if train_x.shape[0] <= self.thresh:
            train_labels[train_index] = self.label_counter
            val_labels[val_index] = self.label_counter
            self.label_counter += 1
            return
        
        # Find best clustering among multiple attempts
        best_val_cluster = None
        min_val_distance = np.inf
        
        for _ in range(self.repeat):
            cluster = MiniBatchKMeans(n_clusters=2, random_state=None, n_init='auto')
            cluster.fit(train_x)
            
            if val_x.shape[0] == 0:
                # No validation samples
                best_val_cluster = cluster
                break
            else:
                # Evaluate clustering quality on validation data
                val_distance = cluster.transform(val_x)
                val_distance = np.min(val_distance, axis=-1)
                val_distance = val_distance.mean()
                
                if val_distance < min_val_distance:
                    min_val_distance = val_distance
                    best_val_cluster = cluster
        
        # Get cluster assignments
        assert best_val_cluster is not None
        train_pred_labels = best_val_cluster.labels_
        train_mask_0 = train_pred_labels == 0
        train_mask_1 = train_pred_labels == 1
        
        # Edge case: all samples in one cluster
        if np.sum(train_mask_0) == 0 or np.sum(train_mask_1) == 0:
            train_labels[train_index] = self.label_counter
            val_labels[val_index] = self.label_counter
            self.label_counter += 1
            return
        
        # Get validation cluster assignments
        if len(val_x) > 0:
            val_pred_labels = best_val_cluster.predict(val_x)
        else:
            val_pred_labels = np.empty([0,], dtype=np.int32)
        
        val_mask_0 = val_pred_labels == 0
        val_mask_1 = val_pred_labels == 1
        
        # Recursively cluster each subset
        self.binary_cluster(
            train_x[train_mask_0], train_index[train_mask_0], train_labels,
            val_x[val_mask_0], val_index[val_mask_0], val_labels
        )
        self.binary_cluster(
            train_x[train_mask_1], train_index[train_mask_1], train_labels,
            val_x[val_mask_1], val_index[val_mask_1], val_labels
        )
    
    def generate_labels(
        self,
        train_representation: np.ndarray,
        val_representation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate cluster labels for all nodes.
        
        Applies binary clustering independently to each node.
        
        Args:
            train_representation: Training representations from Phase 1
                                 Shape: (n_train, hidden_dim, num_nodes, 1)
            val_representation: Validation representations from Phase 1
                               Shape: (n_val, hidden_dim, num_nodes, 1)
                               
        Returns:
            train_labels: Training cluster labels, shape (n_train, num_nodes, 1)
            val_labels: Validation cluster labels, shape (n_val, num_nodes, 1)
        """
        train_num = train_representation.shape[0]
        val_num = val_representation.shape[0]
        
        # Initialize label arrays
        train_labels = np.zeros([train_num, self.num_nodes, 1], dtype=np.int32)
        val_labels = np.zeros([val_num, self.num_nodes, 1], dtype=np.int32)
        
        # Process each node independently
        for node_id in tqdm(range(self.num_nodes), desc="Clustering nodes"):
            # Extract node representations
            # Shape: (n_samples, hidden_dim)
            train_x = train_representation[:, :, node_id, 0]
            val_x = val_representation[:, :, node_id, 0]
            
            # Create index arrays
            train_index = np.arange(train_num)
            val_index = np.arange(val_num)
            
            # Create temporary label arrays for this node
            train_node_labels = np.zeros([train_num], dtype=np.int32)
            val_node_labels = np.zeros([val_num], dtype=np.int32)
            
            # Reset label counter for this node
            self.label_counter = 0
            
            # Apply binary clustering
            self.binary_cluster(
                train_x, train_index, train_node_labels,
                val_x, val_index, val_node_labels
            )
            
            # Store labels
            train_labels[:, node_id, 0] = train_node_labels
            val_labels[:, node_id, 0] = val_node_labels
            
            logger.debug(f"Node {node_id+1}/{self.num_nodes}: {self.label_counter} clusters")
        
        logger.info(f"Label generation complete. Max clusters per node: {np.max(train_labels)+1}")
        
        return train_labels, val_labels


def generate_cluster_labels(
    train_representation_path: str,
    val_representation_path: str,
    output_dir: str,
    num_nodes: int,
    thresh: int = 30,
    repeat: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to generate and save cluster labels.
    
    This function loads representations from Phase 1, generates cluster
    labels using the Labeler, and saves the results.
    
    Args:
        train_representation_path: Path to train_representation.npy
        val_representation_path: Path to val_representation.npy
        output_dir: Directory to save label files
        num_nodes: Number of sensor nodes
        thresh: Minimum cluster size threshold (default: 30)
        repeat: Number of clustering repetitions (default: 10)
        
    Returns:
        Tuple of (train_labels, val_labels) arrays
        
    Example:
        ```python
        generate_cluster_labels(
            train_representation_path="./data/train_representation.npy",
            val_representation_path="./data/val_representation.npy",
            output_dir="./data",
            num_nodes=170,
        )
        ```
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load representations
    logger.info(f"Loading representations...")
    train_repr = np.load(train_representation_path)
    val_repr = np.load(val_representation_path)
    
    logger.info(f"Train representation shape: {train_repr.shape}")
    logger.info(f"Val representation shape: {val_repr.shape}")
    
    # Verify num_nodes
    if train_repr.shape[2] != num_nodes:
        logger.warning(f"num_nodes ({num_nodes}) != representation nodes ({train_repr.shape[2]})")
        num_nodes = train_repr.shape[2]
    
    # Generate labels
    labeler = Labeler(num_nodes=num_nodes, thresh=thresh, repeat=repeat)
    train_labels, val_labels = labeler.generate_labels(train_repr, val_repr)
    
    # Save
    train_path = output_path / "train_cluster_label.npy"
    val_path = output_path / "val_cluster_label.npy"
    
    np.save(train_path, train_labels)
    np.save(val_path, val_labels)
    
    logger.info(f"Saved train_cluster_label.npy: {train_labels.shape}")
    logger.info(f"Saved val_cluster_label.npy: {val_labels.shape}")
    
    return train_labels, val_labels
